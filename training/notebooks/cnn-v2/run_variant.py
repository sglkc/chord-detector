#!/usr/bin/env python3
"""Train one (variant, seed) CNN run as an isolated OS process.

Each run gets a fresh CUDA context (guaranteed by process exit/teardown), which
tf.keras.backend.clear_session() alone does not reliably provide within a single
long-lived process on this machine's 4GB GPU -- consecutive in-process runs were
observed to fail with "Dst tensor is not initialized" after the first run left
fragmented GPU memory behind. Invoked once per run from train-variants.ipynb via
subprocess, so the notebook stays the orchestrating artifact.

Mirrors the architecture/hyperparameters documented in train-variants.ipynb
(batch_size=32 on the RTX 2080 Ti -- see that notebook's title cell for the
batch-size history). Writes one row to variant-results/progress.csv per
success, one row to variant-results/failures.csv per failure, and the full
per-epoch loss/accuracy curve to variant-results/history/{variant}-seed{seed}.csv
for plotting. Safe to re-run: skips (variant, seed) pairs already present in
progress.csv.
"""
import argparse
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from pathlib import Path
import random
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent  # training/
FEATURES_PATH = ROOT / "features" / "training.npz"
OUT_DIR = HERE / "variant-results"
WEIGHTS_DIR = HERE / "weights"
HISTORY_DIR = OUT_DIR / "history"
OUT_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)
PROGRESS_CSV = OUT_DIR / "progress.csv"
FAILURES_CSV = OUT_DIR / "failures.csv"

SAVE_WEIGHTS_SEED = 42
CNN_VALIDATION_SPLIT = 0.1
CNN_TEST_SIZE = 0.1
CNN_EPOCHS = int(os.environ.get("CNN_EPOCHS_OVERRIDE", 50))  # override only for smoke tests
# RTX 2080 Ti (11GB) now available via eGPU -- back to the published batch_size=32,
# no deviation needed. (History: the 4GB laptop 3050 Ti OOM'd at batch=32; batch=16
# hit a real OOM on Adam's moment buffers mid-run; batch=8 avoided OOM but was
# numerically broken -- stuck at chance accuracy from epoch 1, confirmed via an
# isolated controlled test. None of that applies on the 2080 Ti.)
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001
GPU_NAME_PREFERENCE = "2080"  # pin to the eGPU explicitly; TF's device order != nvidia-smi's
FOCUS_CLASSES = ["D_minor_4", "E_diminished_4", "G#_diminished_4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["orig", "clean", "insnorm"])
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()
    variant, seed = args.variant, args.seed

    # Skip if already done (idempotent / resumable)
    if PROGRESS_CSV.exists():
        prev = pd.read_csv(PROGRESS_CSV)
        if len(prev) and ((prev.variant == variant) & (prev.seed == seed)).any():
            print(f"skip {variant}/seed{seed} (already in progress.csv)")
            return 0

    import tensorflow as tf
    from tensorflow.keras import Input, layers, models, optimizers, callbacks
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import precision_recall_fscore_support
    from tensorflow.keras.utils import to_categorical

    physical_devices = tf.config.list_physical_devices('GPU')
    chosen = physical_devices[0] if physical_devices else None
    for gpu in physical_devices:
        name = tf.config.experimental.get_device_details(gpu).get("device_name", "")
        if GPU_NAME_PREFERENCE in name:
            chosen = gpu
            break
    if chosen is not None:
        tf.config.set_visible_devices(chosen, "GPU")
        tf.config.experimental.set_memory_growth(chosen, True)
        print(f"Using GPU: {tf.config.experimental.get_device_details(chosen).get('device_name')}")

    d = np.load(FEATURES_PATH)
    features_all = d["features"]
    labels_all = d["labels"]
    flat = features_all.reshape(len(features_all), -1)
    silent_idx = np.where(np.abs(flat).max(axis=1) == 0)[0]
    clean_idx = np.setdiff1d(np.arange(len(features_all)), silent_idx)
    assert len(silent_idx) == 16 and set(labels_all[silent_idx]) == {"C_diminished_4"}

    idx = np.arange(len(features_all)) if variant == "orig" else clean_idx
    label_encoder = LabelEncoder().fit(labels_all)  # fixed 36-class order, same for every run
    class_names = label_encoder.classes_

    X = features_all[idx]
    y_enc = label_encoder.transform(labels_all[idx])
    y_cat = to_categorical(y_enc, num_classes=36)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=CNN_TEST_SIZE, random_state=seed, stratify=y_cat
    )

    def per_sample_standardize(x):
        m = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        s = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
        return (x - m) / (s + 1e-6)

    norm_layer = (
        layers.Lambda(per_sample_standardize, name="per_sample_standardize")
        if variant == "insnorm"
        else layers.BatchNormalization(axis=1)
    )
    model = models.Sequential([
        Input(shape=(216, 188, 1)),
        norm_layer,
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(36, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, verbose=0),
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0),
    ]
    history = model.fit(
        X_train, y_train, epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
        validation_split=CNN_VALIDATION_SPLIT, callbacks=cbs, verbose=0,
    )

    # Per-epoch curve, saved for human-readable acc/loss plots -- history.history
    # only lives in-process and is otherwise lost once this subprocess exits.
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_df.insert(0, "seed", seed)
    hist_df.insert(0, "variant", variant)
    hist_df.to_csv(HISTORY_DIR / f"{variant}-seed{seed}.csv", index=False)

    y_pred = model.predict(X_test, verbose=0, batch_size=CNN_BATCH_SIZE)
    y_pred_cls = np.argmax(y_pred, axis=1)
    y_true_cls = np.argmax(y_test, axis=1)
    acc = float((y_pred_cls == y_true_cls).mean())
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, labels=np.arange(36), average="macro", zero_division=0
    )
    rec_by_class = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, labels=np.arange(36), average=None, zero_division=0
    )[1]
    focus_recall = {f"recall_{c}": float(rec_by_class[list(class_names).index(c)]) for c in FOCUS_CLASSES}

    row = {
        "variant": variant, "seed": seed,
        "n_train": len(X_train), "n_test": len(X_test),
        "epochs_run": len(history.history["loss"]),
        "test_accuracy": acc, "test_macro_precision": float(p),
        "test_macro_recall": float(r), "test_macro_f1": float(f1),
        **focus_recall,
    }

    if seed == SAVE_WEIGHTS_SEED:
        model.save(WEIGHTS_DIR / f"{variant}-seed{seed}.keras")

    prev = pd.read_csv(PROGRESS_CSV) if PROGRESS_CSV.exists() else pd.DataFrame()
    prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    prev.to_csv(PROGRESS_CSV, index=False)
    print(f"OK {variant}/seed{seed}: {row}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        FAILURES_CSV.parent.mkdir(exist_ok=True)
        prev = pd.read_csv(FAILURES_CSV) if FAILURES_CSV.exists() else pd.DataFrame()
        args = sys.argv
        variant = args[args.index("--variant") + 1] if "--variant" in args else "?"
        seed = args[args.index("--seed") + 1] if "--seed" in args else "?"
        prev = pd.concat([prev, pd.DataFrame([{
            "variant": variant, "seed": seed, "error": f"{type(e).__name__}: {e}"
        }])], ignore_index=True)
        prev.to_csv(FAILURES_CSV, index=False)
        sys.exit(1)
