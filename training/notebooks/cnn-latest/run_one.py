#!/usr/bin/env python3
"""Train one (variant, seed) CNN run as an isolated OS process.

orig  <- training/features/training.npz       (7,200)
clean <- training/features/training-clean.npz (7,172)

Explicit stratified 80/10/10 via two train_test_split calls (not Keras
validation_split, which produced 81/9/10). Split membership is written to
results/splits/{variant}-seed{seed}.csv. Every seed's weights are saved.
"""
import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

from pathlib import Path
import random
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent  # training/
OUT_DIR = HERE / "results"
WEIGHTS_DIR = HERE / "weights"
HISTORY_DIR = OUT_DIR / "history"
SPLITS_DIR = OUT_DIR / "splits"
for d in (OUT_DIR, WEIGHTS_DIR, HISTORY_DIR, SPLITS_DIR):
    d.mkdir(parents=True, exist_ok=True)
PROGRESS_CSV = OUT_DIR / "progress.csv"
FAILURES_CSV = OUT_DIR / "failures.csv"

CNN_VAL_FRACTION_OF_REMAINDER = 1.0 / 9.0  # 10% of pool after holding out 10% test
CNN_TEST_SIZE = 0.1
CNN_EPOCHS = int(os.environ.get("CNN_EPOCHS_OVERRIDE", 50))
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001
FOCUS_CLASSES = ["D_minor_4", "E_diminished_4", "G#_diminished_4"]

VARIANT_NPZ = {
    "orig": ROOT / "features" / "training.npz",
    "clean": ROOT / "features" / "training-clean.npz",
}
AUDIO_DIR = ROOT / "datasets" / "training"


def reconstruct_files(labels):
    files, recon = [], []
    for cls in sorted(p.name for p in AUDIO_DIR.iterdir() if p.is_dir()):
        for name in sorted(p.name for p in (AUDIO_DIR / cls).iterdir() if p.is_file()):
            files.append(f"{cls}/{name}")
            recon.append(cls)
    recon = np.asarray(recon)
    if not np.array_equal(recon, labels):
        raise RuntimeError("dataset walk does not match npz label order")
    return np.asarray(files)


def load_variant(variant):
    path = VARIANT_NPZ[variant]
    assert path.is_file(), f"missing {path}"
    d = np.load(path, allow_pickle=True)
    features = d["features"]
    labels = d["labels"].astype(str)
    if "files" in d.files:
        files = d["files"].astype(str)
    else:
        files = reconstruct_files(labels)
    orig_index = d["orig_index"] if "orig_index" in d.files else np.arange(len(features))
    return features, labels, files, orig_index


def split_80_10_10(n, y_enc, seed):
    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    idx_tv, idx_test = train_test_split(
        idx, test_size=CNN_TEST_SIZE, random_state=seed, stratify=y_enc
    )
    idx_train, idx_val = train_test_split(
        idx_tv,
        test_size=CNN_VAL_FRACTION_OF_REMAINDER,
        random_state=seed,
        stratify=y_enc[idx_tv],
    )
    return idx_train, idx_val, idx_test


def write_split_csv(path, idx_train, idx_val, idx_test, files, labels, orig_index):
    rows = []
    for split, idx in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
        for i in idx:
            rows.append(
                {
                    "source_index": int(i),
                    "orig_index": int(orig_index[i]),
                    "file": files[i],
                    "label": labels[i],
                    "split": split,
                }
            )
    pd.DataFrame(rows).sort_values(["split", "orig_index"]).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["orig", "clean"])
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()
    variant, seed = args.variant, args.seed

    if PROGRESS_CSV.exists():
        prev = pd.read_csv(PROGRESS_CSV)
        if len(prev) and ((prev.variant == variant) & (prev.seed == seed)).any():
            print(f"skip {variant}/seed{seed} (already in progress.csv)")
            return 0

    import tensorflow as tf
    from tensorflow.keras import Input, layers, models, optimizers, callbacks
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import precision_recall_fscore_support
    from tensorflow.keras.utils import to_categorical

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        name = tf.config.experimental.get_device_details(gpus[0]).get("device_name")
        print(f"Using GPU: {name}")

    features, labels, files, orig_index = load_variant(variant)

    orig_all = np.load(VARIANT_NPZ["orig"], allow_pickle=True)
    label_encoder = LabelEncoder().fit(orig_all["labels"].astype(str))
    class_names = label_encoder.classes_
    y_enc = label_encoder.transform(labels)
    y_cat = to_categorical(y_enc, num_classes=36)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    idx_train, idx_val, idx_test = split_80_10_10(len(features), y_enc, seed)
    write_split_csv(
        SPLITS_DIR / f"{variant}-seed{seed}.csv",
        idx_train, idx_val, idx_test, files, labels, orig_index,
    )

    X_train = features[idx_train]
    X_val = features[idx_val]
    X_test = features[idx_test]
    y_train = y_cat[idx_train]
    y_val = y_cat[idx_val]
    y_test = y_cat[idx_test]

    if X_train.ndim == 3:
        X_train = X_train[..., None]
        X_val = X_val[..., None]
        X_test = X_test[..., None]

    model = models.Sequential([
        Input(shape=(216, 188, 1)),
        layers.BatchNormalization(axis=1),
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
        callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=0
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        callbacks=cbs,
        verbose=0,
    )

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
    focus_recall = {
        f"recall_{c}": float(rec_by_class[list(class_names).index(c)])
        for c in FOCUS_CLASSES
    }

    model.save(WEIGHTS_DIR / f"{variant}-seed{seed}.keras")

    row = {
        "variant": variant,
        "seed": seed,
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "epochs_run": int(len(history.history["loss"])),
        "test_accuracy": acc,
        "test_macro_precision": float(p),
        "test_macro_recall": float(r),
        "test_macro_f1": float(f1),
        **focus_recall,
    }
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
        prev = pd.read_csv(FAILURES_CSV) if FAILURES_CSV.exists() else pd.DataFrame()
        argv = sys.argv
        variant = argv[argv.index("--variant") + 1] if "--variant" in argv else "?"
        seed = argv[argv.index("--seed") + 1] if "--seed" in argv else "?"
        prev = pd.concat(
            [prev, pd.DataFrame([{"variant": variant, "seed": seed, "error": f"{type(e).__name__}: {e}"}])],
            ignore_index=True,
        )
        prev.to_csv(FAILURES_CSV, index=False)
        sys.exit(1)
