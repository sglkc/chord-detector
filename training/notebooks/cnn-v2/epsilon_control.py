#!/usr/bin/env python3
"""One-off control: does insnorm/seed43's chance-level training collapse (Table
cross-domain) go away with a larger epsilon in the per-sample standardization layer?

x' = (x - mean(x)) / (std(x) + eps). Collapse was reproduced identically at eps=1e-6
(deterministic, not noise -- see variant-cross-eval.ipynb). This retrains the identical
(variant=insnorm, seed=43) config with eps=1e-3 only, everything else unchanged, and
writes to a SEPARATE file so it never collides with weights_extra/insnorm-seed43.keras
or extra_progress.csv. Mirrors run_variant.py / retrain_extra_weights.py exactly except
for EPS.
"""
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
EPSILON_CONTROL_CSV = OUT_DIR / "epsilon_control.csv"

VARIANT = "insnorm"
SEED = 43
EPS = 1e-3  # vs. 1e-6 in the original insnorm layer
CNN_VALIDATION_SPLIT = 0.1
CNN_TEST_SIZE = 0.1
CNN_EPOCHS = int(os.environ.get("CNN_EPOCHS_OVERRIDE", 50))
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001
GPU_NAME_PREFERENCE = "2080"


def main():
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

    idx = clean_idx  # insnorm always trains on the cleaned set, same as `clean`
    label_encoder = LabelEncoder().fit(labels_all)

    X = features_all[idx]
    y_enc = label_encoder.transform(labels_all[idx])
    y_cat = to_categorical(y_enc, num_classes=36)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=CNN_TEST_SIZE, random_state=SEED, stratify=y_cat
    )

    def per_sample_standardize_eps(x):
        m = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        s = tf.math.reduce_std(x, axis=[1, 2, 3], keepdims=True)
        return (x - m) / (s + EPS)

    model = models.Sequential([
        Input(shape=(216, 188, 1)),
        layers.Lambda(per_sample_standardize_eps, name="per_sample_standardize_eps"),
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
        "variant": f"{VARIANT}-eps{EPS}", "seed": SEED, "eps": EPS,
        "n_train": len(X_train), "n_test": len(X_test),
        "epochs_run": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "test_accuracy": acc, "test_macro_precision": float(p),
        "test_macro_recall": float(r), "test_macro_f1": float(f1),
    }

    model.save(WEIGHTS_DIR / f"insnorm-eps1e-3-seed{SEED}.keras")

    prev = pd.read_csv(EPSILON_CONTROL_CSV) if EPSILON_CONTROL_CSV.exists() else pd.DataFrame()
    prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    prev.to_csv(EPSILON_CONTROL_CSV, index=False)
    print(f"OK {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
