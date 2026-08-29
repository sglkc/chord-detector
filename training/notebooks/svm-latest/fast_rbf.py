#!/usr/bin/env python3
"""BLAS-backed RBF decision function that reproduces sklearn SVC.predict exactly.

sklearn's SVC.predict goes through single-threaded libsvm, which streams the whole
float64 support_vector_ array once per clip. At 40,608 dimensions and ~4,100 support
vectors that is 0.27-0.43 s per clip and memory-bandwidth bound, which makes a
multi-checkpoint evaluation campaign impossible on this machine.

The same decision values fall out of one float32 GEMM per chunk plus 36 small
per-class matmuls. Measured against training/models/svm-best (36 classes, 4143 SVs):
exact label agreement, ~250x faster.

Reconstruction contract, for a fitted SVC(kernel="rbf") with C classes:

  starts = concat([[0], cumsum(n_support_)])
      support_vectors_[starts[c]:starts[c+1]] are the SVs of class classes_[c],
      blocks ordered by classes_ ascending.

  dual_coef_ has shape (C-1, n_SV). Column s lives in the class block containing s.
  Row r of block c is that SV's coefficient in "class c vs the r-th other class",
  where the others are [0..C-1] \\ {c} ascending. Inverting for a pair (i, j), i < j:
      class-i SVs use row j-1, class-j SVs use row i.
  Values are already y_s * alpha_s, so they are summed with no extra sign.

  intercept_ has C(C-1)/2 entries indexed by itertools.combinations(range(C), 2).

  Voting follows libsvm svm_predict_values: dec > 0 votes i, dec <= 0 votes j
  (note the tie goes to j), and the winner is the first maximum in classes_ order,
  which is what np.argmax returns.

Two numerical details that matter, both measured against svm-best rather than bounded
on paper (see gate_svmbest.py):

  - The squared-norm terms are always accumulated in float64, even when the caller
    passes float32 rows. Summing 40,608 squares in float32 loses ~1e-2 on a norm near
    1e5 and shows up as ~1.8e-6 of error in the decision values, which is only ~5x
    below the smallest |decision| actually observed. In float64 that term is exact to
    ~1e-11.
  - float64 is also the default for the cross-term GEMM. float32 there costs 2.4e-7 of
    decision error against a smallest observed |decision| of 8.1e-6 -- a 34x margin,
    which will not survive the ~7.5e7 decision values a full campaign evaluates.
    float64 costs 1.7e-14, a margin of ~5e8, for 2.0 ms/clip against 1.0. Since even
    the slow path is ~95x libsvm and the whole campaign is minutes either way, there
    is nothing to buy by approximating. sv_dtype=np.float32 remains available.
  - The per-class matmuls run under threadpool_limits(1). They are far too small to
    thread and OpenBLAS's launch overhead otherwise costs 20x more than the work.
"""
from __future__ import annotations

import itertools

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
    threadpool_limits = None


class FastRBF:
    """Vectorized replacement for SVC.predict on an RBF kernel.

    Copies everything it needs out of the fitted estimator, so the caller may free
    ``svm.support_vectors_`` afterwards to reclaim the float64 copy.
    """

    def __init__(self, svm, sv_dtype=np.float64, chunk: int = 512):
        classes = np.asarray(svm.classes_)
        n_cls = len(classes)
        sv = np.asarray(svm.support_vectors_)
        dual = np.asarray(svm.dual_coef_)
        n_sv = sv.shape[0]

        if dual.shape != (n_cls - 1, n_sv):
            raise ValueError(f"dual_coef_ is {dual.shape}, expected {(n_cls - 1, n_sv)}")
        starts = np.concatenate([[0], np.cumsum(np.asarray(svm.n_support_))]).astype(int)
        if starts[-1] != n_sv:
            raise ValueError(f"n_support_ sums to {starts[-1]}, expected {n_sv}")
        if getattr(svm, "kernel", None) != "rbf":
            raise ValueError(f"kernel is {svm.kernel!r}, only 'rbf' is supported")

        self.classes_ = classes
        self.n_classes = n_cls
        self.n_sv = n_sv
        self.starts = starts
        self.chunk = int(chunk)
        # _gamma is the resolved float; svm.gamma may still be the string "scale"
        self.gamma = float(getattr(svm, "_gamma", svm.gamma))
        self.sv_dtype = np.dtype(sv_dtype)

        self.sv = np.ascontiguousarray(sv, dtype=self.sv_dtype)
        # norms from the float64 original, never from the downcast copy
        self.sv_sq = np.einsum("ij,ij->i", sv, sv).astype(np.float64)
        self.intercept = np.asarray(svm.intercept_, dtype=np.float64)

        # per-class (n_c, C-1) blocks, contiguous for the small matmuls
        self.dual_T = [
            np.ascontiguousarray(dual[:, starts[c]:starts[c + 1]].T)
            for c in range(n_cls)
        ]

        pairs = np.array(list(itertools.combinations(range(n_cls), 2)))
        self.pair_i, self.pair_j = pairs[:, 0], pairs[:, 1]
        n_pairs = len(pairs)
        if len(self.intercept) != n_pairs:
            raise ValueError(f"intercept_ has {len(self.intercept)}, expected {n_pairs}")

        # branchless vote accumulation: votes = (dec > 0).T @ vsel + jcount
        mi = np.zeros((n_pairs, n_cls), dtype=np.float32)
        mj = np.zeros((n_pairs, n_cls), dtype=np.float32)
        mi[np.arange(n_pairs), self.pair_i] = 1.0
        mj[np.arange(n_pairs), self.pair_j] = 1.0
        self.vsel = mi - mj
        self.jcount = mj.sum(0)

    # -- internals ---------------------------------------------------------

    def _kernel_block(self, xc: np.ndarray) -> np.ndarray:
        """exp(-gamma * ||x - sv||^2) for one chunk, returned float64.

        ``x_sq`` is accumulated in float64 even when the caller hands us float32 rows.
        Summing 40,608 squares in float32 loses ~1e-2 absolute on a norm near 1e5,
        which propagates to ~1e-6 in the decision values -- large enough to sit within
        5x of the smallest observed |decision|. In float64 the same term is exact to
        ~1e-11 and the margin is ~1e5x. The cross term is the cheap one to approximate;
        the norms are not.
        """
        x32 = np.ascontiguousarray(xc, dtype=self.sv_dtype)
        x_sq = np.einsum("ij,ij->i", xc, xc, dtype=np.float64)
        g = x32 @ self.sv.T
        d2 = g.astype(np.float64)
        d2 *= -2.0
        d2 += x_sq[:, None]
        d2 += self.sv_sq[None, :]
        np.maximum(d2, 0.0, out=d2)
        d2 *= -self.gamma
        np.exp(d2, out=d2)
        return d2

    def _decision_from_kernel(self, k: np.ndarray) -> np.ndarray:
        """(n_pairs, m) one-vs-one decision values."""
        m = k.shape[0]
        p = np.empty((self.n_classes, m, self.n_classes - 1), dtype=np.float64)
        ctx = threadpool_limits(limits=1, user_api="blas") if threadpool_limits else None
        if ctx is not None:
            ctx.__enter__()
        try:
            for c in range(self.n_classes):
                np.matmul(k[:, self.starts[c]:self.starts[c + 1]], self.dual_T[c], out=p[c])
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        dec = p[self.pair_i, :, self.pair_j - 1] + p[self.pair_j, :, self.pair_i]
        dec += self.intercept[:, None]
        return dec

    # -- public API --------------------------------------------------------

    def decision_ovo(self, x: np.ndarray) -> np.ndarray:
        """One-vs-one decision values, (n_samples, n_pairs), matching sklearn's order."""
        out = np.empty((len(x), len(self.intercept)), dtype=np.float64)
        for a in range(0, len(x), self.chunk):
            b = min(a + self.chunk, len(x))
            out[a:b] = self._decision_from_kernel(self._kernel_block(x[a:b])).T
        return out

    def predict(self, x: np.ndarray, return_stats: bool = False):
        """Predicted labels, identical to SVC.predict on the same input array.

        With return_stats, also returns per-row max kernel value and min |decision|,
        which are the two numbers that say how far outside the kernel's support the
        input sits and how close the float32 path came to flipping a sign.
        """
        n = len(x)
        pred_idx = np.empty(n, dtype=np.int64)
        k_max = np.empty(n, dtype=np.float64) if return_stats else None
        min_abs_dec = np.inf
        for a in range(0, n, self.chunk):
            b = min(a + self.chunk, n)
            k = self._kernel_block(x[a:b])
            if return_stats:
                k_max[a:b] = k.max(axis=1)
            dec = self._decision_from_kernel(k)
            if return_stats and dec.size:
                min_abs_dec = min(min_abs_dec, float(np.abs(dec).min()))
            votes = (dec > 0).T.astype(np.float32) @ self.vsel
            votes += self.jcount
            pred_idx[a:b] = votes.argmax(axis=1)
        labels = self.classes_[pred_idx]
        if return_stats:
            return labels, {"k_max": k_max, "min_abs_dec": min_abs_dec}
        return labels


def validate_against_sklearn(fast: FastRBF, svm, x32: np.ndarray, tag: str = "") -> dict:
    """Assert exact label agreement with SVC.predict. Raises on any mismatch.

    Both paths must be handed the *same* array object, so that scaler-path precision
    differences cannot be mistaken for a reconstruction bug.
    """
    ref = svm.predict(x32)
    got, stats = fast.predict(x32, return_stats=True)
    mismatches = int((np.asarray(got) != np.asarray(ref)).sum())
    result = {
        "tag": tag,
        "n_checked": int(len(x32)),
        "mismatches": mismatches,
        "min_abs_dec": float(stats["min_abs_dec"]),
        "k_max_mean": float(stats["k_max"].mean()),
        "validated": mismatches == 0,
    }
    if mismatches:
        bad = np.flatnonzero(np.asarray(got) != np.asarray(ref))[:20]
        raise AssertionError(
            f"FastRBF disagreed with SVC.predict on {mismatches}/{len(x32)} rows "
            f"({tag}); first offending row indices {bad.tolist()}, "
            f"min|dec|={result['min_abs_dec']:.3e}. Refusing to continue."
        )
    return result
