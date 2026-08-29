#!/usr/bin/env python3
"""Shared pieces for the svm-latest baseline runs.

The SVM reuses the CNN's persisted partitions from
training/notebooks/cnn-latest/results/splits/ rather than drawing its own. SVC is
deterministic given its data, so the split is the only source of run-to-run variation,
and reusing the CSVs makes every SVM checkpoint paired with a CNN checkpoint instead of
merely matched in protocol.

Three variants:
  orig         pool training.npz (7,200),        partition orig-seed{s}.csv
  clean        pool training-clean.npz (7,172),  partition clean-seed{s}.csv
  clean-fixed  pool training.npz minus the 28 rows in excluded.csv, keeping orig's
               partition. orig vs clean changes the data *and* redraws the split;
               clean-fixed changes only the data, so it isolates the 28 clips.

Fitting uses the train split only, never train+val, so the SVM sees exactly the rows
the CNN trained on. This means svm-latest/orig/seed42 will not reproduce the published
models/svm-best, which was fit on a 90/10 split of the whole pool.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def find_training_root(start: Path | None = None) -> Path:
    """Walk up until the directory holding features/ and models/ is found."""
    start = (start or Path(__file__).parent).resolve()
    for cand in [start, *start.parents]:
        if (cand / "features").is_dir() and (cand / "models").is_dir():
            return cand
        nested = cand / "training"
        if (nested / "features").is_dir() and (nested / "models").is_dir():
            return nested
    raise FileNotFoundError(f"could not locate training root from {start}")


HERE = Path(__file__).parent.resolve()
TRAINING_ROOT = find_training_root(HERE)
FEATURES_DIR = TRAINING_ROOT / "features"
CNN_DIR = TRAINING_ROOT / "notebooks" / "cnn-latest"
CNN_SPLITS = CNN_DIR / "results" / "splits"
EXCLUDED_CSV = CNN_DIR / "results" / "excluded.csv"

OUT_DIR = HERE / "results"
SPLITS_DIR = OUT_DIR / "splits"
CONFUSION_DIR = OUT_DIR / "confusion"
GATE_DIR = OUT_DIR / "gate_failures"
BUNDLE_DIR = HERE / "bundles"

N_CLASSES = 36
FLAT_DIM = 216 * 188
SEEDS = [42, 43, 44, 45, 46]
FOLDS = [0, 1, 2, 3, 4]
OOD_DATASETS = ["thinkpad", "vivo", "flow", "thinkpad-2", "flow-2"]
FOCUS_CLASSES = ["D_minor_4", "E_diminished_4", "G#_diminished_4"]

# Frozen from the published svm-best tuning. Held fixed across every variant and seed
# so the orig/clean comparison isolates the data change rather than a re-tuning.
DEFAULT_C = 1000.0
DEFAULT_GAMMA = 1e-5

VARIANT_SPEC = {
    "orig": {"pool": "training.npz", "split_from": "orig", "drop_excluded": False},
    "clean": {"pool": "training-clean.npz", "split_from": "clean", "drop_excluded": False},
    "clean-fixed": {"pool": "training.npz", "split_from": "orig", "drop_excluded": True},
}
VARIANTS = list(VARIANT_SPEC)


def ensure_dirs() -> None:
    for d in (OUT_DIR, SPLITS_DIR, CONFUSION_DIR, GATE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def class_names() -> np.ndarray:
    """The CNN's class order: sorted unique labels of the orig pool."""
    with np.load(FEATURES_DIR / "training.npz", allow_pickle=True) as z:
        return np.array(sorted(set(z["labels"].astype(str))))


def excluded_orig_index() -> np.ndarray:
    return pd.read_csv(EXCLUDED_CSV)["orig_index"].to_numpy()


def load_split(variant: str, tag: str, write_provenance: bool = True) -> pd.DataFrame:
    """Partition for one run. tag is 'seed42' or 'fold0'.

    Returns the CNN split CSV verbatim for orig/clean. For clean-fixed, drops the rows
    whose orig_index is excluded and writes the realized partition to
    results/splits/clean-fixed-{tag}.csv so the run is auditable.
    """
    spec = VARIANT_SPEC[variant]
    src = CNN_SPLITS / f"{spec['split_from']}-{tag}.csv"
    if not src.is_file():
        raise FileNotFoundError(src)
    df = pd.read_csv(src)
    if spec["drop_excluded"]:
        df = df[~df["orig_index"].isin(excluded_orig_index())].reset_index(drop=True)
        if write_provenance:
            ensure_dirs()
            df.to_csv(SPLITS_DIR / f"{variant}-{tag}.csv", index=False)
    return df


def split_indices(df: pd.DataFrame) -> dict:
    """{'train'|'val'|'test': sorted source_index array}.

    Sorted because libsvm's SMO is row-order sensitive: a fixed order is what makes a
    re-run reproduce the same support set bit for bit.
    """
    return {
        s: np.sort(df.loc[df["split"] == s, "source_index"].to_numpy())
        for s in ("train", "val", "test")
    }


def pool_path(variant: str) -> Path:
    return FEATURES_DIR / VARIANT_SPEC[variant]["pool"]


def fit_scaler_streaming(features, idx, chunk: int = 512):
    """StandardScaler over train rows only, float64 accumulation, no full copy.

    partial_fit is the same Welford update as fit, so this is numerically equivalent to
    fitting on the materialized matrix while peaking at one chunk instead of ~1 GB.
    """
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    for a in range(0, len(idx), chunk):
        blk = features[idx[a:a + chunk]].reshape(-1, FLAT_DIM).astype(np.float64)
        sc.partial_fit(blk)
    return sc


def scaled_matrix(features, idx, scaler, dtype=np.float64, chunk: int = 512):
    """Preallocate and fill scaled rows chunkwise, never materializing an unscaled copy."""
    out = np.empty((len(idx), FLAT_DIM), dtype=dtype)
    mean, scale = scaler.mean_, scaler.scale_
    for a in range(0, len(idx), chunk):
        b = min(a + chunk, len(idx))
        blk = features[idx[a:b]].reshape(-1, FLAT_DIM).astype(np.float64)
        blk -= mean
        blk /= scale
        out[a:b] = blk
    return out


def load_eval_set(dataset: str, scaler, class_to_idx: dict, chunk: int = 512):
    """One recorded eval set as (scaled float32 X, int label array)."""
    with np.load(FEATURES_DIR / f"{dataset}.npz", allow_pickle=True) as z:
        feats = z["features"]
        labels = z["labels"].astype(str)
        x = scaled_matrix(feats, np.arange(len(feats)), scaler, np.float32, chunk)
    y = np.array([class_to_idx[c] for c in labels], dtype=np.int64)
    return x, y


def macro_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(N_CLASSES), average="macro", zero_division=0
    )
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
    }


def per_class_recall(y_true, y_pred) -> np.ndarray:
    from sklearn.metrics import precision_recall_fscore_support

    return precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(N_CLASSES), average=None, zero_division=0
    )[1]


def zero_kernel_sink(intercept: np.ndarray, n_cls: int = N_CLASSES) -> tuple:
    """Class predicted in the K -> 0 limit, from intercept signs alone.

    When every kernel value collapses toward zero each decision reduces to its
    intercept, so the vote vector becomes constant and one class absorbs every input.
    That class is computable before any data is seen, which makes it a prediction the
    empirical sink can be tested against.
    """
    import itertools

    votes = np.zeros(n_cls, dtype=int)
    for p, (i, j) in enumerate(itertools.combinations(range(n_cls), 2)):
        votes[i if intercept[p] > 0 else j] += 1
    order = np.argsort(votes)[::-1]
    return int(order[0]), int(votes[order[0]] - votes[order[1]])


def sink_summary(y_true, y_pred, names: np.ndarray, x_scaled=None, k_max=None) -> dict:
    """Prediction-histogram collapse metrics for one (model, dataset)."""
    n = len(y_pred)
    counts = np.bincount(y_pred, minlength=N_CLASSES)
    share = counts / n
    top = int(np.argmax(counts))
    p = share[share > 0]
    row = {
        "n": int(n),
        "sink_class": str(names[top]),
        "sink_share": float(share[top]),
        "sink_share_excess": float(share[top] - 1.0 / N_CLASSES),
        "top3_share": float(np.sort(share)[::-1][:3].sum()),
        "pred_entropy_bits": float(-(p * np.log2(p)).sum()),
        "n_classes_never_predicted": int((counts == 0).sum()),
    }
    if x_scaled is not None:
        norms = np.linalg.norm(x_scaled.astype(np.float64), axis=1)
        row["scaled_l2_median"] = float(np.median(norms))
        row["scaled_l2_p99"] = float(np.percentile(norms, 99))
        row["scaled_l2_max"] = float(norms.max())
    if k_max is not None:
        row["k_max_mean"] = float(k_max.mean())
        row["k_max_median"] = float(np.median(k_max))
    return row


def quality_root_recall(y_true, y_pred, names: np.ndarray) -> pd.DataFrame:
    """Collapsed recall by chord quality and by root.

    Collapsed, not averaged: a C_major predicted as G_major is a quality hit and a root
    miss. That separates "wrong chord type" from "wrong root", which is the reading we
    want, but it is not the same number as per-class recall averaged within a group.
    """
    roots = np.array([c.split("_")[0] for c in names])
    quals = np.array([c.split("_")[1] for c in names])
    rows = []
    for axis, mapping in (("quality", quals), ("root", roots)):
        t, p = mapping[y_true], mapping[y_pred]
        for g in sorted(set(mapping)):
            m = t == g
            rows.append({
                "axis": axis,
                "group": g,
                "support": int(m.sum()),
                "n_correct": int((p[m] == g).sum()),
                "recall": float((p[m] == g).mean()) if m.any() else 0.0,
            })
    return pd.DataFrame(rows)


def append_rows(path: Path, rows, drop_keys: dict | None = None) -> None:
    """Read-modify-append-rewrite, matching the cnn-latest style.

    drop_keys removes matching rows first so --force re-runs never duplicate.
    """
    ensure_dirs()
    new = pd.DataFrame(rows if isinstance(rows, list) else [rows])
    if path.exists():
        prev = pd.read_csv(path)
        if drop_keys and len(prev):
            mask = np.ones(len(prev), dtype=bool)
            for k, v in drop_keys.items():
                if k in prev.columns:
                    mask &= prev[k].astype(str) == str(v)
            prev = prev[~mask]
        new = pd.concat([prev, new], ignore_index=True)
    new.to_csv(path, index=False)


def already_done(path: Path, **keys) -> bool:
    if not path.exists():
        return False
    df = pd.read_csv(path)
    if not len(df):
        return False
    mask = np.ones(len(df), dtype=bool)
    for k, v in keys.items():
        if k not in df.columns:
            return False
        mask &= df[k].astype(str) == str(v)
    return bool(mask.any())
