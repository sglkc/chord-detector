#!/usr/bin/env python3
"""Score the tuned cnn-revised checkpoint on recorded OOD sets and eval-only overlays.

Single checkpoint only: variant="clean", seed=42 (the promoted winner of
tune.ipynb's search, at weights/clean-seed42.keras). Adapted from
cnn-latest/run_eval.py with the multi-variant/multi-seed loop stripped out and
paths repointed at this folder; output schema is identical so downstream
notebooks (onset-classify.ipynb, cnn-svm-comparison.ipynb) key off the same
column names and confusion-matrix filename convention.

Writes results/ood_seeds.csv and results/overlay_seeds.csv, plus
per-(dataset[, overlay]) confusion matrices under results/confusion/.
"""
import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent
FEATURES_DIR = ROOT / "features"
WEIGHTS_DIR = HERE / "weights"
OUT_DIR = HERE / "results"
CM_DIR = OUT_DIR / "confusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)

VARIANT = "clean"
SEED = 42
OOD_DATASETS = ["thinkpad", "vivo", "flow", "thinkpad-2", "flow-2"]
OVERLAY_SUFFIXES = {
    "noise": "noise-interior_domestic",
    "rir": "rir-indoor_no_bathroom",
    "dir": "dir-micirp",
}
OVERLAY_FACTOR = {
    "noise": "environment",
    "rir": "environment",
    "dir": "recording-setting",
}
BATCH_SIZE = 32
OOD_CSV = OUT_DIR / "ood_seeds.csv"
OVERLAY_CSV = OUT_DIR / "overlay_seeds.csv"
RECALL_CSV = OUT_DIR / "per_class_recall_seeds.csv"
OVERLAY_RECALL_CSV = OUT_DIR / "overlay_per_class_recall.csv"
PROGRESS_CSV = OUT_DIR / "progress.csv"


def class_order():
    labels = np.load(FEATURES_DIR / "training.npz", allow_pickle=True)["labels"].astype(str)
    return np.array(sorted(np.unique(labels)))


def load_xy(path, class_to_idx):
    d = np.load(path, allow_pickle=True)
    X = d["features"].astype(np.float32)
    if X.ndim == 3:
        X = X[..., None]
    y = np.array([class_to_idx[v] for v in d["labels"].astype(str)])
    return X, y


def score(model, X, y, class_names):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

    pred = np.argmax(model.predict(X, verbose=0, batch_size=BATCH_SIZE), axis=1)
    acc = float((pred == y).mean())
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, labels=np.arange(len(class_names)), average="macro", zero_division=0
    )
    rec = precision_recall_fscore_support(
        y, pred, labels=np.arange(len(class_names)), average=None, zero_division=0
    )[1]
    cm = confusion_matrix(y, pred, labels=np.arange(len(class_names)))
    return acc, float(p), float(r), float(f1), rec, cm, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ood", "overlay", "both"], default="both")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow.keras.models import load_model

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        name = tf.config.experimental.get_device_details(gpus[0]).get("device_name")
        print(f"Using GPU: {name}")

    class_names = class_order()
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    wp = WEIGHTS_DIR / f"{VARIANT}-seed{SEED}.keras"
    assert wp.is_file(), f"missing {wp} -- run tune.ipynb's promotion cell first"
    model = tf.keras.models.load_model(wp)

    if args.stage in ("ood", "both"):
        ood_rows, rec_rows = [], []
        for ds in OOD_DATASETS:
            X, y = load_xy(FEATURES_DIR / f"{ds}.npz", class_to_idx)
            acc, p, r, f1, rec, cm, _ = score(model, X, y, class_names)
            ood_rows.append({
                "variant": VARIANT, "seed": SEED, "dataset": ds,
                "accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1,
            })
            for cls, rv in zip(class_names, rec):
                rec_rows.append({
                    "variant": VARIANT, "seed": SEED, "dataset": ds,
                    "class": cls, "recall": float(rv),
                })
            pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
                CM_DIR / f"{ds}_{VARIANT}_seed{SEED}.csv"
            )
            del X, y
        pd.DataFrame(ood_rows).to_csv(OOD_CSV, index=False)
        pd.DataFrame(rec_rows).to_csv(RECALL_CSV, index=False)
        print(f"OOD: {len(ood_rows)} datasets -> {OOD_CSV}")

        # in-domain test split, same schema as cnn-latest/results/progress.csv
        from sklearn.metrics import precision_recall_fscore_support

        split = pd.read_csv(HERE.parent / "cnn-latest" / "results" / "splits" / "clean-seed42.csv")
        idx_test = split.loc[split.split == "test", "source_index"].to_numpy(dtype=int)
        d = np.load(FEATURES_DIR / "training-clean.npz", allow_pickle=True)
        X_test = d["features"][idx_test].astype(np.float32)
        if X_test.ndim == 3:
            X_test = X_test[..., None]
        y_test = np.array([class_to_idx[v] for v in d["labels"].astype(str)[idx_test]])
        acc, p, r, f1, rec, cm, _ = score(model, X_test, y_test, class_names)
        pd.DataFrame([{
            "variant": VARIANT, "seed": SEED,
            "n_train": int((split.split == "train").sum()),
            "n_val": int((split.split == "val").sum()),
            "n_test": int(len(idx_test)),
            "test_accuracy": acc, "test_macro_precision": p,
            "test_macro_recall": r, "test_macro_f1": f1,
        }]).to_csv(PROGRESS_CSV, index=False)
        print(f"in-domain test: acc={acc:.4f} -> {PROGRESS_CSV}")

    if args.stage in ("overlay", "both"):
        overlay_rows, rec_rows = [], []
        for ds in OOD_DATASETS:
            for overlay, suffix in OVERLAY_SUFFIXES.items():
                X, y = load_xy(FEATURES_DIR / f"{ds}-{suffix}.npz", class_to_idx)
                acc, p, r, f1, rec, cm, _ = score(model, X, y, class_names)
                overlay_rows.append({
                    "variant": VARIANT, "seed": SEED, "dataset": ds, "overlay": overlay,
                    "factor": OVERLAY_FACTOR[overlay],
                    "accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1,
                })
                pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
                    CM_DIR / f"{ds}_{overlay}_{VARIANT}_seed{SEED}.csv"
                )
                rec_rows.extend({
                    "variant": VARIANT, "seed": SEED, "dataset": ds, "overlay": overlay,
                    "class": cls, "recall": float(rv),
                } for cls, rv in zip(class_names, rec))
                del X, y
        pd.DataFrame(overlay_rows).to_csv(OVERLAY_CSV, index=False)
        pd.DataFrame(rec_rows).to_csv(OVERLAY_RECALL_CSV, index=False)
        print(f"overlay: {len(overlay_rows)} cells -> {OVERLAY_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
