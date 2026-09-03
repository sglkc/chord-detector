#!/usr/bin/env python3
"""One hyperparameter-search trial as an isolated OS process.

Architecture is frozen (Table 3.5): identical Sequential to
`cnn-latest/run_one.py`. Only training hyperparameters vary (Table 3.6 rows +
the two callback settings). Split and init are NOT part of the search: every
trial reuses the exact partition `cnn-latest/results/splits/clean-seed42.csv`
already wrote (the same partition the SVM's clean/seed-42 fit used, so nothing
needs re-pairing) and seed 42 throughout, so trials differ only by
hyperparameters.

Selection metric is mean(A1, A2, B1, B2) OOD accuracy, computed directly each
trial from the four recorded generalization datasets
(flow=A1, flow-2=A2, thinkpad=B1, thinkpad-2=B2). This is explicit selection
on the reported test scenarios, not a held-out dev set -- state that plainly
wherever these results are written up.

Writes one row to results/tuning_trials.csv (append, replacing any existing
row for the same trial number), results/history/trial{NN}.csv, and
weights/trial{NN}.keras.
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
ROOT = HERE.parent.parent  # training/
CNN_LATEST = HERE.parent / "cnn-latest"
FEATURES_DIR = ROOT / "features"
WEIGHTS_DIR = HERE / "weights"
OUT_DIR = HERE / "results"
HISTORY_DIR = OUT_DIR / "history"
SPLIT_CSV = CNN_LATEST / "results" / "splits" / "clean-seed42.csv"
TRIALS_CSV = OUT_DIR / "tuning_trials.csv"
FAILURES_CSV = OUT_DIR / "tuning_failures.csv"
for d in (WEIGHTS_DIR, HISTORY_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
FEATURES_NPZ = FEATURES_DIR / "training-clean.npz"
OOD_DATASETS = ["flow", "flow-2", "thinkpad", "thinkpad-2"]  # A1, A2, B1, B2
OOD_LABEL = {"flow": "A1", "flow-2": "A2", "thinkpad": "B1", "thinkpad-2": "B2"}
CNN_EPOCHS = int(os.environ.get("CNN_EPOCHS_OVERRIDE", 50))
BATCH_SIZE_EVAL = 32  # fixed batch size used for scoring, independent of trial's training batch size


def load_split():
    assert SPLIT_CSV.is_file(), f"missing {SPLIT_CSV}"
    df = pd.read_csv(SPLIT_CSV)
    idx = {}
    for split in ("train", "val", "test"):
        idx[split] = df.loc[df.split == split, "source_index"].to_numpy(dtype=int)
    return idx["train"], idx["val"], idx["test"]


def load_xy(path, class_to_idx):
    d = np.load(path, allow_pickle=True)
    X = d["features"].astype(np.float32)
    if X.ndim == 3:
        X = X[..., None]
    y = np.array([class_to_idx[v] for v in d["labels"].astype(str)])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True, type=int)
    ap.add_argument("--lr", required=True, type=float)
    ap.add_argument("--batch-size", required=True, type=int)
    ap.add_argument("--es-patience", type=int, default=6)
    ap.add_argument("--rlr-patience", type=int, default=2)
    ap.add_argument("--rlr-factor", type=float, default=0.1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    trial = args.trial

    hist_path = HISTORY_DIR / f"trial{trial:02d}.csv"
    if hist_path.exists() and not args.force:
        print(f"skip trial {trial} (history exists; pass --force to retrain)")
        return 0

    trial_start = datetime.now().replace(microsecond=0)

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

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    d = np.load(FEATURES_NPZ, allow_pickle=True)
    features = d["features"]
    labels = d["labels"].astype(str)

    orig_all = np.load(FEATURES_DIR / "training.npz", allow_pickle=True)
    class_names = np.array(sorted(np.unique(orig_all["labels"].astype(str))))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    label_encoder = LabelEncoder().fit(class_names)
    y_enc = label_encoder.transform(labels)
    y_cat = to_categorical(y_enc, num_classes=36)

    idx_train, idx_val, idx_test = load_split()

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
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=args.rlr_factor, patience=args.rlr_patience, verbose=0
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=args.es_patience, restore_best_weights=True, verbose=0
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CNN_EPOCHS,
        batch_size=args.batch_size,
        callbacks=cbs,
        verbose=0,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_df.insert(0, "trial", trial)
    hist_df.to_csv(hist_path, index=False)

    def eval_split(X, y):
        loss, acc = model.evaluate(X, y, batch_size=BATCH_SIZE_EVAL, verbose=0)
        return float(loss), float(acc)

    val_loss, val_acc = eval_split(X_val, y_val)
    test_loss, test_acc = eval_split(X_test, y_test)

    y_pred_test = np.argmax(model.predict(X_test, verbose=0, batch_size=BATCH_SIZE_EVAL), axis=1)
    y_true_test = np.argmax(y_test, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_test, y_pred_test, labels=np.arange(36), average="macro", zero_division=0
    )

    ood_acc = {}
    for ds in OOD_DATASETS:
        X, y = load_xy(FEATURES_DIR / f"{ds}.npz", class_to_idx)
        pred = np.argmax(model.predict(X, verbose=0, batch_size=BATCH_SIZE_EVAL), axis=1)
        ood_acc[ds] = float((pred == y).mean())
        del X, y

    mean_ood = float(np.mean([ood_acc[ds] for ds in OOD_DATASETS]))

    model.save(WEIGHTS_DIR / f"trial{trial:02d}.keras")

    minutes = (datetime.now().replace(microsecond=0) - trial_start).total_seconds() / 60.0
    row = {
        "trial": trial,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "es_patience": args.es_patience,
        "rlr_patience": args.rlr_patience,
        "rlr_factor": args.rlr_factor,
        "epochs_run": int(len(history.history["loss"])),
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_macro_precision": float(p),
        "test_macro_recall": float(r),
        "test_macro_f1": float(f1),
        "A1": ood_acc["flow"],
        "A2": ood_acc["flow-2"],
        "B1": ood_acc["thinkpad"],
        "B2": ood_acc["thinkpad-2"],
        "mean_ood": mean_ood,
        "minutes": minutes,
    }
    prev = pd.read_csv(TRIALS_CSV) if TRIALS_CSV.exists() else pd.DataFrame()
    if len(prev):
        prev = prev[prev.trial != trial]
    prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    prev = prev.sort_values("trial")
    prev.to_csv(TRIALS_CSV, index=False)

    print(f"OK trial {trial}: mean_ood={mean_ood:.4f} " + " ".join(f"{OOD_LABEL[k]}={v:.4f}" for k, v in ood_acc.items()))
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
        prev = pd.concat(
            [prev, pd.DataFrame([{"trial": trial, "error": f"{type(e).__name__}: {e}"}])],
            ignore_index=True,
        )
        prev.to_csv(FAILURES_CSV, index=False)
        sys.exit(1)
