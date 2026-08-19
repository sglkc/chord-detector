#!/usr/bin/env python3
"""One (variant, fold) of 5-fold CV as an isolated OS process.

Outer StratifiedKFold(n=5, seed=42) holds out 20% as test. From the remaining
80%, a stratified 1/8 split is val, so the fold is 70/10/20 of the pool.
5-fold cannot be 80/10/10 because each test fold is 20%. Split membership is
written to results/splits/{variant}-fold{k}.csv. Fold weights are not saved.
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
ROOT = HERE.parent.parent
OUT_DIR = HERE / "results"
SPLITS_DIR = OUT_DIR / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
KFOLD_CSV = OUT_DIR / "kfold.csv"
FAILURES_CSV = OUT_DIR / "kfold_failures.csv"

CNN_EPOCHS = int(os.environ.get("CNN_EPOCHS_OVERRIDE", 50))
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001
N_SPLITS = 5
VAL_FRACTION_OF_TRAINVAL = 0.125  # 10% of the full pool

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
    ap.add_argument("--fold", required=True, type=int, help="0-based fold index")
    args = ap.parse_args()
    variant, fold = args.variant, args.fold
    if not 0 <= fold < N_SPLITS:
        raise ValueError(f"fold must be in 0..{N_SPLITS - 1}")

    if KFOLD_CSV.exists():
        prev = pd.read_csv(KFOLD_CSV)
        if len(prev) and ((prev.variant == variant) & (prev.fold == fold)).any():
            print(f"skip {variant}/fold{fold} (already in kfold.csv)")
            return 0

    import tensorflow as tf
    from tensorflow.keras import Input, layers, models, optimizers, callbacks
    from sklearn.model_selection import StratifiedKFold, train_test_split
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
    y_enc = label_encoder.transform(labels)
    y_cat = to_categorical(y_enc, num_classes=36)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(y_enc)), y_enc))
    idx_tv, idx_test = splits[fold]
    seed = 42 + fold
    idx_train, idx_val = train_test_split(
        idx_tv,
        test_size=VAL_FRACTION_OF_TRAINVAL,
        random_state=seed,
        stratify=y_enc[idx_tv],
    )
    write_split_csv(
        SPLITS_DIR / f"{variant}-fold{fold}.csv",
        idx_train, idx_val, idx_test, files, labels, orig_index,
    )

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, X_val, X_test = features[idx_train], features[idx_val], features[idx_test]
    if X_train.ndim == 3:
        X_train = X_train[..., None]
        X_val = X_val[..., None]
        X_test = X_test[..., None]
    y_train, y_val, y_test = y_cat[idx_train], y_cat[idx_val], y_cat[idx_test]

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

    y_pred = np.argmax(model.predict(X_test, verbose=0, batch_size=CNN_BATCH_SIZE), axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = float((y_pred == y_true).mean())
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(36), average="macro", zero_division=0
    )
    row = {
        "variant": variant,
        "fold": fold,
        "seed": seed,
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "epochs_run": int(len(history.history["loss"])),
        "test_accuracy": acc,
        "test_macro_precision": float(p),
        "test_macro_recall": float(r),
        "test_macro_f1": float(f1),
    }
    prev = pd.read_csv(KFOLD_CSV) if KFOLD_CSV.exists() else pd.DataFrame()
    prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    prev.to_csv(KFOLD_CSV, index=False)
    print(f"OK {variant}/fold{fold}: {row}")
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
        fold = argv[argv.index("--fold") + 1] if "--fold" in argv else "?"
        prev = pd.concat(
            [prev, pd.DataFrame([{"variant": variant, "fold": fold, "error": f"{type(e).__name__}: {e}"}])],
            ignore_index=True,
        )
        prev.to_csv(FAILURES_CSV, index=False)
        sys.exit(1)
