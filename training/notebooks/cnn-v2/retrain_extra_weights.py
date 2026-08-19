#!/usr/bin/env python3
"""Retrain a specific (variant, seed) with weights forced to save, into a
SEPARATE dir (weights_extra/) and a SEPARATE log (variant-results/extra_progress.csv)
-- does not touch progress.csv or weights/ from the main Phase-2 batch.

One-off investigation tool: used to check whether the `clean`/seed42 OOD
collapse found in variant-cross-eval.ipynb is seed-42-specific bad luck or
systematic to the `clean` variant. Mirrors run_variant.py's architecture and
training logic exactly.
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
WEIGHTS_DIR = HERE / "weights_extra"
OUT_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)
EXTRA_PROGRESS_CSV = OUT_DIR / "extra_progress.csv"

CNN_VALIDATION_SPLIT = 0.1
CNN_TEST_SIZE = 0.1
CNN_EPOCHS = int(os.environ.get("CNN_EPOCHS_OVERRIDE", 50))
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001
GPU_NAME_PREFERENCE = "2080"
FOCUS_CLASSES = ["D_minor_4", "E_diminished_4", "G#_diminished_4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["orig", "clean", "insnorm"])
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()
    variant, seed = args.variant, args.seed

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
    label_encoder = LabelEncoder().fit(labels_all)
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

    y_pred = model.predict(X_test, verbose=0, batch_size=CNN_BATCH_SIZE)
    y_pred_cls = np.argmax(y_pred, axis=1)
    y_true_cls = np.argmax(y_test, axis=1)
    acc = float((y_pred_cls == y_true_cls).mean())
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, labels=np.arange(36), average="macro", zero_division=0
    )

    row = {
        "variant": variant, "seed": seed,
        "n_train": len(X_train), "n_test": len(X_test),
        "epochs_run": len(history.history["loss"]),
        "test_accuracy": acc, "test_macro_precision": float(p),
        "test_macro_recall": float(r), "test_macro_f1": float(f1),
    }

    model.save(WEIGHTS_DIR / f"{variant}-seed{seed}.keras")

    prev = pd.read_csv(EXTRA_PROGRESS_CSV) if EXTRA_PROGRESS_CSV.exists() else pd.DataFrame()
    prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    prev.to_csv(EXTRA_PROGRESS_CSV, index=False)
    print(f"OK {variant}/seed{seed}: {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
