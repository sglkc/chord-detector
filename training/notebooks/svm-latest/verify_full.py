#!/usr/bin/env python3
"""Full-coverage verification: every clip of every recorded set through stock libsvm.

The per-model gate in run_one.py samples 128 rows, roughly 1.6% of what a checkpoint
actually evaluates. This closes that gap on a real model at the real dimensionality
(36 classes, 4143 support vectors, 40,608 features) by running 100% of the 7,200
recorded clips through sklearn's own SVC.predict and comparing every label.

It uses the published models/svm-best, so nothing has to be refitted. That model is
the same architecture, kernel, and hyperparameters as every checkpoint in the sweep,
so a clean result here covers the reconstruction for all of them.

Slow by design: ~0.2 s/clip through libsvm, about 25 minutes for all five sets. Run it
when nothing else is competing for memory bandwidth -- two processes each holding a
1.35 GB support-vector array degrade libsvm by roughly 20x.

Run:  python verify_full.py                    # all five recorded sets
      python verify_full.py --datasets thinkpad-2 --limit 400
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import svm_common as C  # noqa: E402
from fast_rbf import FastRBF  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=C.OOD_DATASETS)
    ap.add_argument("--limit", type=int, default=0, help="clips per dataset, 0 = all")
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    import joblib

    mdir = C.TRAINING_ROOT / "models" / "svm-best"
    svm = joblib.load(mdir / "model.pkl")
    scaler = joblib.load(mdir / "scaler.pkl")
    fast = FastRBF(svm, chunk=args.chunk)
    print(f"model: {len(svm.classes_)} classes, {int(svm.n_support_.sum())} SVs, "
          f"{svm.support_vectors_.shape[1]} features, gamma {svm._gamma:g}\n")

    rows = []
    grand_n = grand_bad = 0
    for ds in args.datasets:
        with np.load(C.FEATURES_DIR / f"{ds}.npz", allow_pickle=True) as z:
            feats = z["features"]
            if args.limit:
                feats = feats[: args.limit]
            n = len(feats)
            x = C.scaled_matrix(feats, np.arange(n), scaler, np.float32, args.chunk)
        del feats

        t = time.time()
        got = fast.predict(x)
        fast_s = time.time() - t

        # stock sklearn, chunked only so progress is visible; identical to one call
        t = time.time()
        ref = np.empty(n, dtype=got.dtype)
        for a in range(0, n, args.chunk):
            b = min(a + args.chunk, n)
            ref[a:b] = svm.predict(x[a:b])
            print(f"  {ds} {b}/{n} libsvm rows ({time.time() - t:.0f}s)", flush=True)
        ref_s = time.time() - t

        bad = int((got != ref).sum())
        grand_n += n
        grand_bad += bad
        rows.append({"dataset": ds, "n": n, "mismatches": bad,
                     "fast_s": round(fast_s, 2), "libsvm_s": round(ref_s, 1),
                     "speedup": round(ref_s / max(fast_s, 1e-9), 1)})
        print(f"{ds:11s} {n - bad}/{n} identical   fast {fast_s:.1f}s   "
              f"libsvm {ref_s:.0f}s   {ref_s / max(fast_s, 1e-9):.0f}x\n", flush=True)
        del x

    df = pd.DataFrame(rows)
    C.ensure_dirs()
    df.to_csv(C.OUT_DIR / "verify_full.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nTOTAL {grand_n - grand_bad}/{grand_n} predictions identical to sklearn SVC.predict")
    if grand_bad:
        print("VERIFICATION FAILED")
        return 1
    print("FULL-COVERAGE VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
