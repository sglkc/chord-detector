#!/usr/bin/env python3
"""One (trial-config, fold) of 5-fold CV, in-domain only, as an isolated OS process.

Rescoring fix: the first tuning pass (`run_tune.py` / `tune.ipynb`) selected its
winner by mean(A1, A2, B1, B2) -- the four reported generalization scenarios --
which is a worse leak than the SVM's own tuning ever had (the SVM's search never
touched anything outside `training-clean.npz`). This script re-scores the exact
same 10 (lr, batch_size) configs Optuna already drew, this time using only
5-fold CV on the training pool, mirroring `svm-tuning.ipynb`'s
`StratifiedKFold(n_splits=5)` objective exactly. No OOD dataset is touched here.

Reuses the fold assignments already written by `cnn-latest/run_kfold.py`
(`cnn-latest/results/splits/clean-fold{0..4}.csv`) rather than redrawing them --
same folds the published baseline's own k-fold CV and the SVM's `kfold.ipynb`
already used, so nothing needs re-pairing.

Per-fold models are not persisted (matches svm-tuning.ipynb: fit, score,
discard). The winning config's *already-trained* single-split checkpoint
(`weights/trial{NN}.keras`, trained during the first tuning pass on the fixed
`clean-seed42` split) is what gets promoted -- this script only decides which
one that is.

Writes one row to results/cv_scores.csv per (trial, fold).
"""
import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

from pathlib import Path
from datetime import datetime
import random
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent
CNN_LATEST = HERE.parent / "cnn-latest"
FEATURES_NPZ = ROOT / "features" / "training-clean.npz"
OUT_DIR = HERE / "results"
CV_SCORES_CSV = OUT_DIR / "cv_scores.csv"
TRIALS_CSV = OUT_DIR / "tuning_trials.csv"
FAILURES_CSV = OUT_DIR / "cv_failures.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
CNN_EPOCHS = int(os.environ.get("CNN_EPOCHS_OVERRIDE", 50))
ES_PATIENCE = 6
RLR_PATIENCE = 2
RLR_FACTOR = 0.1


def load_fold_split(fold):
    path = CNN_LATEST / "results" / "splits" / f"clean-fold{fold}.csv"
    assert path.is_file(), f"missing {path}"
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
    ap.add_argument("--trial", required=True, type=int, help="row in tuning_trials.csv to look up lr/batch_size")
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    trial, fold = args.trial, args.fold
    if not 0 <= fold < N_FOLDS:
        raise ValueError(f"fold must be in 0..{N_FOLDS - 1}")

    if CV_SCORES_CSV.exists() and not args.force:
        prev = pd.read_csv(CV_SCORES_CSV)
        if len(prev) and ((prev.trial == trial) & (prev.fold == fold)).any():
            print(f"skip trial{trial}/fold{fold} (already in cv_scores.csv)")
            return 0

    tuned = pd.read_csv(TRIALS_CSV)
    row0 = tuned.loc[tuned.trial == trial].iloc[0]
    lr, batch_size = float(row0.lr), int(row0.batch_size)

    trial_start = datetime.now().replace(microsecond=0)

    import tensorflow as tf
    from tensorflow.keras import Input, layers, models, optimizers, callbacks
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    seed = 42 + fold
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    d = np.load(FEATURES_NPZ, allow_pickle=True)
    features = d["features"]
    labels = d["labels"].astype(str)

    orig_all = np.load(ROOT / "features" / "training.npz", allow_pickle=True)
    class_names = np.array(sorted(np.unique(orig_all["labels"].astype(str))))
    label_encoder = LabelEncoder().fit(class_names)
    y_enc = label_encoder.transform(labels)
    y_cat = to_categorical(y_enc, num_classes=36)

    idx_train, idx_val, idx_test = load_fold_split(fold)
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
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=RLR_FACTOR, patience=RLR_PATIENCE, verbose=0),
        callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True, verbose=0),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CNN_EPOCHS,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=0,
    )

    def eval_split(X, y):
        loss, acc = model.evaluate(X, y, batch_size=32, verbose=0)
        return float(loss), float(acc)

    val_loss, val_acc = eval_split(X_val, y_val)
    test_loss, test_acc = eval_split(X_test, y_test)

    minutes = (datetime.now().replace(microsecond=0) - trial_start).total_seconds() / 60.0
    row = {
        "trial": trial,
        "fold": fold,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "epochs_run": int(len(history.history["loss"])),
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "minutes": minutes,
    }
    replace_row(CV_SCORES_CSV, row, {"trial": trial, "fold": fold})
    print(f"OK trial{trial}/fold{fold}: test_loss={test_loss:.6f} test_acc={test_acc:.4f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback

        traceback.print_exc()
        prev = pd.read_csv(FAILURES_CSV) if FAILURES_CSV.exists() else pd.DataFrame()
        argv = sys.argv
        trial = argv[argv.index("--trial") + 1] if "--trial" in argv else "?"
        fold = argv[argv.index("--fold") + 1] if "--fold" in argv else "?"
        prev = pd.concat(
            [prev, pd.DataFrame([{"trial": trial, "fold": fold, "error": f"{type(e).__name__}: {e}"}])],
            ignore_index=True,
        )
        prev.to_csv(FAILURES_CSV, index=False)
        sys.exit(1)
