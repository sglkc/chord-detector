#!/usr/bin/env python3
"""Property tests for the FastRBF reconstruction.

The gate inside run_one.py checks 128 rows of the real model. That catches gross
breakage but samples ~1.6% of what gets evaluated, and it exercises exactly one
(n_classes, gamma, class-balance) configuration. These tests sweep the structural
assumptions instead, on models small enough that stock SVC.predict is affordable for
100% of rows:

  - the support-vector block ordering implied by n_support_ follows classes_
  - the dual_coef_ row convention other(c, r) = r if r < c else r + 1
  - the intercept_ pair order from itertools.combinations
  - libsvm's tie-break (first maximum in classes_ order), which np.argmax must match

Each is checked against both SVC.predict (labels) and SVC.decision_function with
decision_function_shape='ovo' (continuous values, a far stronger signal than labels).

Run:  python test_fast_rbf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from fast_rbf import FastRBF  # noqa: E402

CASES = [
    # (n_classes, n_per_class, n_features, gamma, label_kind, shuffle, unbalanced)
    (3, 20, 40, 0.01, "int", False, False),
    (5, 15, 60, 0.001, "str", False, False),
    (5, 15, 60, 0.001, "str", True, False),
    (10, 12, 80, 1e-4, "int", True, False),
    (10, 12, 80, "scale", "str", True, True),
    (36, 8, 120, 1e-5, "str", True, False),
    (36, 8, 120, 1e-5, "str", True, True),
    (4, 30, 200, 1e-6, "int", True, False),
    (7, 25, 50, 0.05, "str", True, True),
]


def make_case(n_cls, n_per, n_feat, label_kind, shuffle, unbalanced, rng):
    centers = rng.normal(scale=3.0, size=(n_cls, n_feat))
    xs, ys = [], []
    for c in range(n_cls):
        k = n_per if not unbalanced else max(4, n_per - c * (n_per // (2 * n_cls) + 1))
        xs.append(centers[c] + rng.normal(scale=1.0, size=(k, n_feat)))
        ys.append(np.full(k, c))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    if shuffle:
        p = rng.permutation(len(x))
        x, y = x[p], y[p]
    if label_kind == "str":
        y = np.array([f"class_{v:02d}" for v in y])
    return x, y


def main() -> int:
    from sklearn.svm import SVC

    total_rows = total_ties = 0
    worst_dec_err = 0.0
    failures = []

    for i, (n_cls, n_per, n_feat, gamma, kind, shuffle, unbal) in enumerate(CASES):
        rng = np.random.default_rng(1000 + i)
        x, y = make_case(n_cls, n_per, n_feat, kind, shuffle, unbal, rng)
        x_te, _ = make_case(n_cls, 12, n_feat, kind, True, False, rng)
        # push some test rows far outside the training manifold, where kernel values
        # collapse and the vote vector is most likely to hit ties
        x_te = np.vstack([x_te, x_te[:8] * 6.0, rng.normal(scale=12.0, size=(8, n_feat))])

        svm = SVC(kernel="rbf", C=10.0, gamma=gamma).fit(x, y)
        fast = FastRBF(svm)

        got = fast.predict(x_te)
        ref = svm.predict(x_te)
        n_bad = int((np.asarray(got) != np.asarray(ref)).sum())

        shape = svm.decision_function_shape
        svm.decision_function_shape = "ovo"
        ref_dec = svm.decision_function(x_te)
        svm.decision_function_shape = shape
        dec_err = float(np.abs(fast.decision_ovo(x_te) - ref_dec).max())
        worst_dec_err = max(worst_dec_err, dec_err)

        # count rows whose top vote is shared, i.e. where the tie-break decides
        votes = np.zeros((len(x_te), n_cls), dtype=int)
        for p, (a, b) in enumerate(zip(fast.pair_i, fast.pair_j)):
            votes[ref_dec[:, p] > 0, a] += 1
            votes[ref_dec[:, p] <= 0, b] += 1
        ties = int((votes == votes.max(1, keepdims=True)).sum(1).__gt__(1).sum())

        total_rows += len(x_te)
        total_ties += ties
        status = "ok" if n_bad == 0 else f"FAIL {n_bad}"
        if n_bad:
            failures.append((i, n_bad))
        print(f"case {i}: {n_cls:2d} classes, {len(x):3d} train, {int(svm.n_support_.sum()):3d} SV, "
              f"gamma={gamma!s:>6}, {'str' if kind == 'str' else 'int'} labels, "
              f"{'shuffled' if shuffle else 'sorted  '}, "
              f"{'unbalanced' if unbal else 'balanced  '} -> "
              f"{len(x_te) - n_bad}/{len(x_te)} labels, dec err {dec_err:.2e} [{status}]")

    print(f"\n{total_rows} test rows across {len(CASES)} fits, {total_ties} decided by tie-break")
    print(f"worst one-vs-one decision error: {worst_dec_err:.3e}")
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    print("ALL PROPERTY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
