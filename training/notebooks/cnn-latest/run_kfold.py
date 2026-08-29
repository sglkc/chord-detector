#!/usr/bin/env python3
"""One (variant, fold) of 5-fold CV as an isolated OS process.

Outer StratifiedKFold(n=5, seed=42) holds out 20% as test. From the remaining
80%, a stratified 1/8 split is val, so the fold is 70/10/20 of the pool.
5-fold cannot be 80/10/10 because each test fold is 20%. Split membership is
written to results/splits/{variant}-fold{k}.csv. Fold weights are not saved.

History is written to results/history/{variant}-fold{k}.csv. Restored-weight
train / val / test loss and accuracy go to results/kfold_scores.csv. The
published results/kfold.csv is not overwritten.
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
HISTORY_DIR = OUT_DIR / "history"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
KFOLD_CSV = OUT_DIR / "kfold.csv"
SCORES_CSV = OUT_DIR / "kfold_scores.csv"
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


def load_persisted_split(path):
    df = pd.read_csv(path)
    idx = {}
    for split in ("train", "val", "test"):
        idx[split] = df.loc[df.split == split, "source_index"].to_numpy(dtype=int)
    return idx["train"], idx["val"], idx["test"]


def replace_row(path, row, keys):
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(prev):
        mask = pd.Series(True, index=prev.index)
        for key, value in keys.items():
            mask &= prev[key] == value
        prev = prev.loc[~mask]
    prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    prev.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["orig", "clean"])
    ap.add_argument("--fold", required=True, type=int, help="0-based fold index")
    ap.add_argument(
        "--force",
        action="store_true",
        help="retrain even if results/history/{variant}-fold{k}.csv already exists",
    )
    args = ap.parse_args()
    variant, fold = args.variant, args.fold
    if not 0 <= fold < N_SPLITS:
        raise ValueError(f"fold must be in 0..{N_SPLITS - 1}")

    hist_path = HISTORY_DIR / f"{variant}-fold{fold}.csv"
    if hist_path.exists() and not args.force:
        print(f"skip {variant}/fold{fold} (history exists; pass --force to retrain)")
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

    seed = 42 + fold
    split_path = SPLITS_DIR / f"{variant}-fold{fold}.csv"
    if split_path.exists():
        idx_train, idx_val, idx_test = load_persisted_split(split_path)
        print(f"reusing {split_path.name}", flush=True)
    else:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        splits = list(skf.split(np.zeros(len(y_enc)), y_enc))
        idx_tv, idx_test = splits[fold]
        idx_train, idx_val = train_test_split(
            idx_tv,
            test_size=VAL_FRACTION_OF_TRAINVAL,
            random_state=seed,
            stratify=y_enc[idx_tv],
        )
        write_split_csv(
            split_path, idx_train, idx_val, idx_test, files, labels, orig_index,
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

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_df.insert(0, "fold", fold)
    hist_df.insert(0, "seed", seed)
    hist_df.insert(0, "variant", variant)
    hist_df.to_csv(hist_path, index=False)

    def eval_split(X, y):
        loss, acc = model.evaluate(X, y, batch_size=CNN_BATCH_SIZE, verbose=0)
        return float(loss), float(acc)

    train_loss, train_acc = eval_split(X_train, y_train)
    val_loss, val_acc = eval_split(X_val, y_val)
    test_loss, test_acc = eval_split(X_test, y_test)

    y_pred = np.argmax(model.predict(X_test, verbose=0, batch_size=CNN_BATCH_SIZE), axis=1)
    y_true = np.argmax(y_test, axis=1)
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
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_precision": float(p),
        "test_macro_recall": float(r),
        "test_macro_f1": float(f1),
    }
    replace_row(SCORES_CSV, row, {"variant": variant, "fold": fold})

    if KFOLD_CSV.exists():
        pub = pd.read_csv(KFOLD_CSV)
        hit = pub[(pub.variant == variant) & (pub.fold == fold)]
        if len(hit):
            old = float(hit.test_accuracy.iloc[0])
            print(
                f"published test_accuracy {old:.6f}  rerun {test_acc:.6f}  "
                f"delta {test_acc - old:+.6f}",
                flush=True,
            )

    print(f"OK {variant}/fold{fold}: {row}", flush=True)
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
