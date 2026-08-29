#!/usr/bin/env python3
"""Score one (variant, seed) SVM on eval-only overlay CQTs.

Refits the same way as run_one.py (no bundle is kept on disk) and writes
results/overlay_seeds.csv. Overlay files already exist under training/features/.

Run:  python run_overlay.py --variant clean --seed 42
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import svm_common as C  # noqa: E402
from fast_rbf import FastRBF  # noqa: E402

OVERLAY_CSV = C.OUT_DIR / "overlay_seeds.csv"
OVERLAY_RECALL_CSV = C.OUT_DIR / "overlay_per_class_recall.csv"
FAILURES_CSV = C.OUT_DIR / "overlay_failures.csv"

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
OVERLAYS = list(OVERLAY_SUFFIXES)


def overlay_npz(dataset: str, overlay: str) -> Path:
    return C.FEATURES_DIR / f"{dataset}-{OVERLAY_SUFFIXES[overlay]}.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=C.VARIANTS)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--C", dest="c_val", type=float, default=C.DEFAULT_C)
    ap.add_argument("--gamma", type=float, default=C.DEFAULT_GAMMA)
    ap.add_argument("--cache-size", type=int, default=int(os.environ.get("SVM_CACHE_SIZE_MB", 512)))
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("SVM_CHUNK", 512)))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    variant, seed, chunk = args.variant, args.seed, args.chunk

    C.ensure_dirs()
    if not args.force and C.already_done(OVERLAY_CSV, variant=variant, seed=seed):
        print(f"skip {variant}/seed{seed} (already in overlay_seeds.csv)")
        return 0

    missing = [
        overlay_npz(ds, ov)
        for ds in C.OOD_DATASETS
        for ov in OVERLAYS
        if not overlay_npz(ds, ov).is_file()
    ]
    if missing:
        raise FileNotFoundError("missing overlay features:\n" + "\n".join(str(p) for p in missing))

    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    names = C.class_names()
    class_to_idx = {c: i for i, c in enumerate(names)}
    df = C.load_split(variant, f"seed{seed}")
    idx = C.split_indices(df)
    y_train = np.array(
        [class_to_idx[c] for c in
         df.loc[df.split == "train"].sort_values("source_index")["label"]],
        dtype=np.int64,
    )
    print(f"{variant}/seed{seed}: train {len(idx['train'])}", flush=True)

    z = np.load(C.pool_path(variant), allow_pickle=True)
    feats = z["features"]
    scaler = C.fit_scaler_streaming(feats, idx["train"], chunk)
    x_train = C.scaled_matrix(feats, idx["train"], scaler, np.float64, chunk)
    del feats
    z.close()
    del z
    gc.collect()

    t0 = time.time()
    svm = SVC(kernel="rbf", C=args.c_val, gamma=args.gamma,
              cache_size=args.cache_size, random_state=seed)
    svm.fit(x_train, y_train)
    print(f"  fit {time.time() - t0:.1f}s  nSV {int(svm.n_support_.sum())}", flush=True)
    fast = FastRBF(svm, sv_dtype=np.float64, chunk=chunk)
    svm.support_vectors_ = None
    del svm, x_train
    gc.collect()

    rows, rec_rows = [], []
    for ds in C.OOD_DATASETS:
        for ov in OVERLAYS:
            path = overlay_npz(ds, ov)
            with np.load(path, allow_pickle=True) as z:
                feats = z["features"]
                labels = z["labels"].astype(str)
                x = C.scaled_matrix(feats, np.arange(len(feats)), scaler, np.float32, chunk)
            y = np.array([class_to_idx[c] for c in labels], dtype=np.int64)
            pred = np.asarray(fast.predict(x), dtype=np.int64)
            m = C.macro_metrics(y, pred)
            rec = C.per_class_recall(y, pred)
            cm = confusion_matrix(y, pred, labels=np.arange(C.N_CLASSES))
            pd.DataFrame(cm, index=names, columns=names).to_csv(
                C.CONFUSION_DIR / f"{ds}_{ov}_{variant}_seed{seed}.csv"
            )
            rec_rows.extend(
                {
                    "variant": variant, "seed": seed, "dataset": ds,
                    "overlay": ov, "class": names[i], "recall": float(rec[i]),
                }
                for i in range(C.N_CLASSES)
            )
            rows.append({
                "variant": variant, "seed": seed, "dataset": ds,
                "overlay": ov, "factor": OVERLAY_FACTOR[ov], **m,
            })
            print(f"  {ds:11s} {ov:5s} acc {m['accuracy']:.4f}", flush=True)
            del x, y, pred
            gc.collect()

    C.append_rows(OVERLAY_CSV, rows, drop_keys={"variant": variant, "seed": seed})
    C.append_rows(OVERLAY_RECALL_CSV, rec_rows, drop_keys={"variant": variant, "seed": seed})
    print(f"OK {variant}/seed{seed}  {len(rows)} overlay cells", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        argv = sys.argv
        C.ensure_dirs()
        C.append_rows(FAILURES_CSV, {
            "variant": argv[argv.index("--variant") + 1] if "--variant" in argv else "?",
            "seed": argv[argv.index("--seed") + 1] if "--seed" in argv else "?",
            "error": f"{type(exc).__name__}: {exc}",
        })
        sys.exit(1)
