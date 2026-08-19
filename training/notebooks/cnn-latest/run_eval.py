#!/usr/bin/env python3
"""Score every cnn-latest checkpoint on recorded OOD sets and eval-only overlays.

Resume-friendly. Writes results/ood_seeds.csv and results/overlay_seeds.csv,
plus per-(dataset, variant, seed) confusion matrices.
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

OOD_DATASETS = ["thinkpad", "vivo", "flow", "thinkpad-2", "flow-2"]
VARIANTS = ["orig", "clean"]
SEEDS = [42, 43, 44, 45, 46]
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


def already_done(csv_path, variant, seed, extra_cols=None):
    if not csv_path.exists():
        return False
    prev = pd.read_csv(csv_path)
    mask = (prev.variant == variant) & (prev.seed == seed)
    if extra_cols:
        for k, v in extra_cols.items():
            mask &= prev[k] == v
    return bool(mask.any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ood", "overlay", "both"], default="both")
    ap.add_argument("--variant", choices=VARIANTS, default=None)
    ap.add_argument("--seed", type=int, default=None)
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
    variants = [args.variant] if args.variant else VARIANTS
    seeds = [args.seed] if args.seed is not None else SEEDS

    ood_cache = {}
    if args.stage in ("ood", "both"):
        for ds in OOD_DATASETS:
            ood_cache[ds] = load_xy(FEATURES_DIR / f"{ds}.npz", class_to_idx)

    for variant in variants:
        for seed in seeds:
            wp = WEIGHTS_DIR / f"{variant}-seed{seed}.keras"
            if not wp.exists():
                print(f"missing {wp}, skip")
                continue
            model = load_model(wp)

            if args.stage in ("ood", "both") and not already_done(OOD_CSV, variant, seed):
                ood_rows, rec_rows = [], []
                for ds in OOD_DATASETS:
                    X, y = ood_cache[ds]
                    acc, p, r, f1, rec, cm, _ = score(model, X, y, class_names)
                    ood_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "dataset": ds,
                            "accuracy": acc,
                            "macro_precision": p,
                            "macro_recall": r,
                            "macro_f1": f1,
                        }
                    )
                    for cls, rv in zip(class_names, rec):
                        rec_rows.append(
                            {
                                "variant": variant,
                                "seed": seed,
                                "dataset": ds,
                                "class": cls,
                                "recall": float(rv),
                            }
                        )
                    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
                        CM_DIR / f"{ds}_{variant}_seed{seed}.csv"
                    )
                prev = pd.read_csv(OOD_CSV) if OOD_CSV.exists() else pd.DataFrame()
                prev = pd.concat([prev, pd.DataFrame(ood_rows)], ignore_index=True)
                prev.to_csv(OOD_CSV, index=False)
                prev_r = pd.read_csv(RECALL_CSV) if RECALL_CSV.exists() else pd.DataFrame()
                prev_r = pd.concat([prev_r, pd.DataFrame(rec_rows)], ignore_index=True)
                prev_r.to_csv(RECALL_CSV, index=False)
                print(f"OOD {variant}/seed{seed}: {len(ood_rows)} datasets")

            if args.stage in ("overlay", "both") and not already_done(OVERLAY_CSV, variant, seed):
                overlay_rows = []
                for ds in OOD_DATASETS:
                    for overlay, suffix in OVERLAY_SUFFIXES.items():
                        X, y = load_xy(FEATURES_DIR / f"{ds}-{suffix}.npz", class_to_idx)
                        acc, p, r, f1, _, _, _ = score(model, X, y, class_names)
                        overlay_rows.append(
                            {
                                "variant": variant,
                                "seed": seed,
                                "dataset": ds,
                                "overlay": overlay,
                                "factor": OVERLAY_FACTOR[overlay],
                                "accuracy": acc,
                                "macro_precision": p,
                                "macro_recall": r,
                                "macro_f1": f1,
                            }
                        )
                        del X, y
                prev = pd.read_csv(OVERLAY_CSV) if OVERLAY_CSV.exists() else pd.DataFrame()
                prev = pd.concat([prev, pd.DataFrame(overlay_rows)], ignore_index=True)
                prev.to_csv(OVERLAY_CSV, index=False)
                print(f"overlay {variant}/seed{seed}: {len(overlay_rows)} cells")

            del model
            tf.keras.backend.clear_session()

    return 0


if __name__ == "__main__":
    sys.exit(main())
