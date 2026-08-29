#!/usr/bin/env python3
"""Step zero: prove FastRBF reproduces SVC.predict on the published svm-best model.

Runs in its own process because the fitted estimator holds a 1.35 GB float64
support-vector array. Compares on the hardest available rows: `vivo`, whose scaled row
norms are ~1.7x the training set's, so kernel values are smallest and decision values
closest to a sign flip.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import svm_common as C  # noqa: E402
from fast_rbf import FastRBF, validate_against_sklearn  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--dataset", default="vivo")
    args = ap.parse_args()

    import joblib

    mdir = C.TRAINING_ROOT / "models" / "svm-best"
    svm = joblib.load(mdir / "model.pkl")
    scaler = joblib.load(mdir / "scaler.pkl")
    print(f"svm-best: {len(svm.classes_)} classes, {int(svm.n_support_.sum())} SVs, "
          f"gamma {svm._gamma:g}, C {svm.C:g}")

    with np.load(C.FEATURES_DIR / f"{args.dataset}.npz", allow_pickle=True) as z:
        feats = z["features"][: args.n]
    x = C.scaled_matrix(feats, np.arange(len(feats)), scaler, np.float32)
    del feats

    t = time.time()
    _ = svm.predict(x[:8])
    ref_ms = (time.time() - t) * 1000 / 8

    # Reference one-vs-one decision values straight from libsvm, to measure the error
    # rather than bound it. A label flips only when the error exceeds |decision|, so the
    # number that matters is min|dec| / max|error|.
    shape = svm.decision_function_shape
    svm.decision_function_shape = "ovo"
    ref_dec = svm.decision_function(x[:8])
    svm.decision_function_shape = shape

    print(f"libsvm         {ref_ms:.0f} ms/clip")
    print(f"mean max K     "
          f"{FastRBF(svm, sv_dtype=np.float64).predict(x[:8], return_stats=True)[1]['k_max'].mean():.4f}")

    ok = True
    for dt, name in ((np.float32, "f32_gemm"), (np.float64, "f64_gemm")):
        fast = FastRBF(svm, sv_dtype=dt)
        res = validate_against_sklearn(fast, svm, x, tag=f"svm-best/{args.dataset}/{name}")
        err = np.abs(fast.decision_ovo(x[:8]) - ref_dec)
        margin = res["min_abs_dec"] / max(err.max(), 1e-300)
        t = time.time()
        fast.predict(x)
        ms = (time.time() - t) * 1000 / len(x)
        print(f"\n{name}")
        print(f"  agreement    {res['n_checked'] - res['mismatches']}/{res['n_checked']} exact")
        print(f"  dec error    max {err.max():.3e}  median {np.median(err):.3e}")
        print(f"  min |dec|    {res['min_abs_dec']:.3e}  -> margin {margin:.3g}x")
        print(f"  speed        {ms:.2f} ms/clip  ({ref_ms / ms:.0f}x libsvm)")
        ok &= res["validated"]
        del fast

    print("\nGATE PASSED" if ok else "\nGATE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
