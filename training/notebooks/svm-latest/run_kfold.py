#!/usr/bin/env python3
"""One (variant, fold) of the 5-fold in-domain cross-validation.

Reuses cnn-latest's persisted fold partitions (70/10/20, fold seed 42+fold), so the
SVM and CNN k-fold tables are computed on identical rows. In-domain only, matching
cnn-latest/results/kfold.csv. No model is saved.

Run:  python run_kfold.py --variant orig --fold 0
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
from fast_rbf import FastRBF, validate_against_sklearn  # noqa: E402

KFOLD_CSV = C.OUT_DIR / "kfold.csv"
FAILURES_CSV = C.OUT_DIR / "kfold_failures.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=C.VARIANTS)
    ap.add_argument("--fold", required=True, type=int, choices=C.FOLDS)
    ap.add_argument("--C", dest="c_val", type=float, default=C.DEFAULT_C)
    ap.add_argument("--gamma", type=float, default=C.DEFAULT_GAMMA)
    ap.add_argument("--cache-size", type=int, default=int(os.environ.get("SVM_CACHE_SIZE_MB", 512)))
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("SVM_CHUNK", 512)))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    variant, fold, chunk = args.variant, args.fold, args.chunk
    seed = 42 + fold  # cnn-latest convention

    C.ensure_dirs()
    if not args.force and C.already_done(KFOLD_CSV, variant=variant, fold=fold):
        print(f"skip {variant}/fold{fold} (already in kfold.csv)")
        return 0

    from sklearn.svm import SVC

    names = C.class_names()
    class_to_idx = {c: i for i, c in enumerate(names)}

    df = C.load_split(variant, f"fold{fold}")
    idx = C.split_indices(df)
    y = {s: np.array([class_to_idx[c] for c in
                      df.loc[df.split == s].sort_values("source_index")["label"]],
                     dtype=np.int64)
         for s in ("train", "val", "test")}
    print(f"{variant}/fold{fold}: train {len(idx['train'])} val {len(idx['val'])} "
          f"test {len(idx['test'])}", flush=True)

    z = np.load(C.pool_path(variant), allow_pickle=True)
    feats = z["features"]
    scaler = C.fit_scaler_streaming(feats, idx["train"], chunk)
    x_train = C.scaled_matrix(feats, idx["train"], scaler, np.float64, chunk)
    x_test = C.scaled_matrix(feats, idx["test"], scaler, np.float32, chunk)
    del feats
    z.close()
    del z
    gc.collect()

    t0 = time.time()
    svm = SVC(kernel="rbf", C=args.c_val, gamma=args.gamma,
              cache_size=args.cache_size, random_state=seed)
    svm.fit(x_train, y["train"])
    fit_s = time.time() - t0

    fast = FastRBF(svm, sv_dtype=np.float64, chunk=chunk)
    rng = np.random.default_rng(seed)
    gate_x = np.ascontiguousarray(
        x_test[rng.choice(len(x_test), size=min(128, len(x_test)), replace=False)],
        dtype=np.float32)
    gate = validate_against_sklearn(fast, svm, gate_x, tag=f"{variant}-fold{fold}")
    print(f"  fit {fit_s:.1f}s  nSV {int(svm.n_support_.sum())}  "
          f"gate OK {gate['n_checked']}", flush=True)

    svm.support_vectors_ = None
    del svm, x_train
    gc.collect()

    pred = np.asarray(fast.predict(x_test), dtype=np.int64)
    m = C.macro_metrics(y["test"], pred)

    C.append_rows(KFOLD_CSV, {
        "variant": variant, "fold": fold, "seed": seed,
        "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])),
        "n_test": int(len(idx["test"])), "epochs_run": pd.NA,
        "test_accuracy": m["accuracy"],
        "test_macro_precision": m["macro_precision"],
        "test_macro_recall": m["macro_recall"],
        "test_macro_f1": m["macro_f1"],
    }, drop_keys={"variant": variant, "fold": fold})
    print(f"OK {variant}/fold{fold}: acc {m['accuracy']:.4f}", flush=True)
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
            "fold": argv[argv.index("--fold") + 1] if "--fold" in argv else "?",
            "error": f"{type(exc).__name__}: {exc}",
        })
        sys.exit(1)
