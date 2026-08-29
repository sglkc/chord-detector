#!/usr/bin/env python3
"""Fit and evaluate one (variant, seed) SVM as an isolated OS process.

Unlike cnn-latest, fitting and evaluation happen in the same process. A Keras
checkpoint is ~124 MB; a fitted SVC on this data is ~1.35 GB, so persisting 15 of them
to re-load later costs 20 GB and minutes per load under this machine's memory pressure.
The process boundary is still per-run, which is what actually returns memory to the OS.

Run:  python run_one.py --variant orig --seed 42
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import svm_common as C  # noqa: E402
from fast_rbf import FastRBF, validate_against_sklearn  # noqa: E402

PROGRESS_CSV = C.OUT_DIR / "progress.csv"
FAILURES_CSV = C.OUT_DIR / "failures.csv"
OOD_CSV = C.OUT_DIR / "ood_seeds.csv"
RECALL_CSV = C.OUT_DIR / "per_class_recall_seeds.csv"
DIAG_CSV = C.OUT_DIR / "diagnostics.csv"
SUPPORT_CSV = C.OUT_DIR / "support_seeds.csv"
PRED_HIST_CSV = C.OUT_DIR / "pred_hist_seeds.csv"
SINK_CSV = C.OUT_DIR / "sink_seeds.csv"
QR_CSV = C.OUT_DIR / "quality_root_recall_seeds.csv"

DERIVED = [OOD_CSV, RECALL_CSV, DIAG_CSV, SUPPORT_CSV, PRED_HIST_CSV, SINK_CSV, QR_CSV]


def drop_previous(variant: str, seed: int) -> None:
    for path in [PROGRESS_CSV, *DERIVED]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not len(df) or "variant" not in df.columns:
            continue
        keep = ~((df.variant.astype(str) == variant) & (df.seed.astype(str) == str(seed)))
        df[keep].to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=C.VARIANTS)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--C", dest="c_val", type=float, default=C.DEFAULT_C)
    ap.add_argument("--gamma", type=float, default=C.DEFAULT_GAMMA)
    ap.add_argument("--cache-size", type=int, default=int(os.environ.get("SVM_CACHE_SIZE_MB", 512)))
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("SVM_CHUNK", 512)))
    ap.add_argument("--gate-n", type=int, default=128)
    ap.add_argument("--save-bundle", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    variant, seed, chunk = args.variant, args.seed, args.chunk

    C.ensure_dirs()
    if args.force:
        drop_previous(variant, seed)
    elif C.already_done(PROGRESS_CSV, variant=variant, seed=seed):
        print(f"skip {variant}/seed{seed} (already in progress.csv)")
        return 0

    import joblib
    import sklearn
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    names = C.class_names()
    class_to_idx = {c: i for i, c in enumerate(names)}

    df = C.load_split(variant, f"seed{seed}")
    idx = C.split_indices(df)
    y_all = {s: np.array([class_to_idx[c] for c in
                          df.loc[df.split == s].sort_values("source_index")["label"]],
                         dtype=np.int64)
             for s in ("train", "val", "test")}
    print(f"{variant}/seed{seed}: train {len(idx['train'])} val {len(idx['val'])} "
          f"test {len(idx['test'])}", flush=True)

    # ---- features: scaler on train rows only, then materialize scaled matrices ----
    z = np.load(C.pool_path(variant), allow_pickle=True)
    feats = z["features"]
    t0 = time.time()
    scaler = C.fit_scaler_streaming(feats, idx["train"], chunk)
    x_train = C.scaled_matrix(feats, idx["train"], scaler, np.float64, chunk)
    x_val = C.scaled_matrix(feats, idx["val"], scaler, np.float32, chunk)
    x_test = C.scaled_matrix(feats, idx["test"], scaler, np.float32, chunk)
    del feats
    z.close()
    del z
    gc.collect()
    prep_s = time.time() - t0

    train_norms = np.linalg.norm(x_train, axis=1)

    # ---- fit ----
    t0 = time.time()
    svm = SVC(kernel="rbf", C=args.c_val, gamma=args.gamma,
              cache_size=args.cache_size, random_state=seed)
    svm.fit(x_train, y_all["train"])
    fit_s = time.time() - t0
    print(f"  fit {fit_s:.1f}s  nSV {int(svm.n_support_.sum())}", flush=True)

    fast = FastRBF(svm, sv_dtype=np.float64, chunk=chunk)

    # ---- validation gate: must reproduce SVC.predict exactly, or stop ----
    rng = np.random.default_rng(seed)
    gate_half = max(1, args.gate_n // 2)
    gate_rows = [x_test[rng.choice(len(x_test), size=min(gate_half, len(x_test)), replace=False)]]
    vivo_x, vivo_y = C.load_eval_set("vivo", scaler, class_to_idx, chunk)
    gate_rows.append(vivo_x[rng.choice(len(vivo_x), size=gate_half, replace=False)])
    gate_x = np.ascontiguousarray(np.vstack(gate_rows), dtype=np.float32)
    t0 = time.time()
    gate = validate_against_sklearn(fast, svm, gate_x, tag=f"{variant}-seed{seed}")
    print(f"  gate OK {gate['n_checked']} rows, min|dec| {gate['min_abs_dec']:.3e} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # diagnostics that need the estimator, before freeing it
    dual = np.asarray(svm.dual_coef_)
    n_sv = np.asarray(svm.n_support_)
    bounded = int((np.abs(dual) >= args.c_val - 1e-9).sum())
    sink_zero, sink_zero_margin = C.zero_kernel_sink(np.asarray(svm.intercept_))
    dual_abs_max = float(np.abs(dual).max())
    dual_abs_sum_median = float(np.median(np.abs(dual).sum(axis=1)))

    if args.save_bundle:
        C.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(C.BUNDLE_DIR / f"{variant}-seed{seed}.npz",
                 sv=fast.sv, dual_coef=dual, intercept=np.asarray(svm.intercept_),
                 n_support=n_sv, classes=names, gamma=fast.gamma,
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        joblib.dump(scaler, C.BUNDLE_DIR / f"{variant}-seed{seed}-scaler.pkl")

    svm.support_vectors_ = None
    del svm, x_train, dual
    gc.collect()

    # ---- evaluate ----
    ood_rows, recall_rows, hist_rows, sink_rows, qr_rows = [], [], [], [], []
    predict_ms = []
    recall_by_tag: dict[str, np.ndarray] = {}

    def score(tag: str, x, y, write_confusion: bool):
        t = time.time()
        pred_lbl, stats = fast.predict(x, return_stats=True)
        predict_ms.append((time.time() - t) * 1000.0 / max(1, len(x)))
        pred = np.asarray(pred_lbl, dtype=np.int64)
        m = C.macro_metrics(y, pred)
        rec = C.per_class_recall(y, pred)
        recall_by_tag[tag] = rec
        if write_confusion:
            cm = confusion_matrix(y, pred, labels=np.arange(C.N_CLASSES))
            pd.DataFrame(cm, index=names, columns=names).to_csv(
                C.CONFUSION_DIR / f"{tag}_{variant}_seed{seed}.csv")
            recall_rows.extend({"variant": variant, "seed": seed, "dataset": tag,
                                "class": names[i], "recall": float(rec[i])}
                               for i in range(C.N_CLASSES))
        counts = np.bincount(pred, minlength=C.N_CLASSES)
        true_counts = np.bincount(y, minlength=C.N_CLASSES)
        hist_rows.extend({"variant": variant, "seed": seed, "dataset": tag,
                          "class": names[i], "n_pred": int(counts[i]),
                          "pred_share": float(counts[i] / len(y)),
                          "n_true": int(true_counts[i]),
                          "true_share": float(true_counts[i] / len(y))}
                         for i in range(C.N_CLASSES))
        s = C.sink_summary(y, pred, names, x_scaled=x, k_max=stats["k_max"])
        s.update({"variant": variant, "seed": seed, "dataset": tag,
                  "sink_matches_zero_kernel_class": bool(
                      s["sink_class"] == str(names[sink_zero]))})
        sink_rows.append(s)
        qr = C.quality_root_recall(y, pred, names)
        qr.insert(0, "dataset", tag)
        qr.insert(0, "seed", seed)
        qr.insert(0, "variant", variant)
        qr_rows.append(qr)
        return m

    m_test = score("test", x_test, y_all["test"], write_confusion=False)
    m_val = score("val", x_val, y_all["val"], write_confusion=False)
    del x_test, x_val
    gc.collect()

    for ds in C.OOD_DATASETS:
        if ds == "vivo":
            x, y = vivo_x, vivo_y
        else:
            x, y = C.load_eval_set(ds, scaler, class_to_idx, chunk)
        m = score(ds, x, y, write_confusion=True)
        ood_rows.append({"variant": variant, "seed": seed, "dataset": ds, **m})
        print(f"  {ds:11s} acc {m['accuracy']:.4f}  sink {sink_rows[-1]['sink_class']} "
              f"{sink_rows[-1]['sink_share'] * 100:.1f}%", flush=True)
        del x, y
        if ds == "vivo":
            del vivo_x, vivo_y
        gc.collect()

    # ---- write ----
    keys = {"variant": variant, "seed": seed}
    C.append_rows(PROGRESS_CSV, {
        **keys,
        "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])),
        "n_test": int(len(idx["test"])), "epochs_run": pd.NA,
        "test_accuracy": m_test["accuracy"],
        "test_macro_precision": m_test["macro_precision"],
        "test_macro_recall": m_test["macro_recall"],
        "test_macro_f1": m_test["macro_f1"],
        **{f"recall_{c}": float(recall_by_tag["test"][list(names).index(c)])
           for c in C.FOCUS_CLASSES},
    }, drop_keys=keys)
    C.append_rows(OOD_CSV, ood_rows, drop_keys=keys)
    C.append_rows(RECALL_CSV, recall_rows, drop_keys=keys)
    C.append_rows(SUPPORT_CSV, [
        {**keys, "class": names[i], "n_train_class": int((y_all["train"] == i).sum()),
         "n_support": int(n_sv[i]),
         "sv_fraction": float(n_sv[i] / max(1, (y_all["train"] == i).sum()))}
        for i in range(C.N_CLASSES)], drop_keys=keys)
    C.append_rows(PRED_HIST_CSV, hist_rows, drop_keys=keys)
    C.append_rows(SINK_CSV, sink_rows, drop_keys=keys)
    C.append_rows(QR_CSV, pd.concat(qr_rows, ignore_index=True).to_dict("records"),
                  drop_keys=keys)
    C.append_rows(DIAG_CSV, {
        **keys, "fold": pd.NA,
        "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])),
        "n_test": int(len(idx["test"])),
        "kernel": "rbf", "C": args.c_val, "gamma": args.gamma,
        "cache_size_mb": args.cache_size, "chunk": chunk,
        "prep_seconds": prep_s, "fit_seconds": fit_s,
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_sv_total": int(n_sv.sum()),
        "sv_fraction": float(n_sv.sum() / len(idx["train"])),
        "n_sv_min_class": int(n_sv.min()), "n_sv_max_class": int(n_sv.max()),
        "n_sv_median_class": float(np.median(n_sv)),
        "n_sv_bounded": bounded, "bounded_fraction": float(bounded / max(1, n_sv.sum())),
        "dual_coef_abs_max": dual_abs_max,
        "dual_coef_abs_sum_median": dual_abs_sum_median,
        "train_l2_median": float(np.median(train_norms)),
        "train_l2_p99": float(np.percentile(train_norms, 99)),
        "train_l2_max": float(train_norms.max()),
        "train_l2_min": float(train_norms.min()),
        "sink_class_zero_kernel": str(names[sink_zero]),
        "sink_class_zero_kernel_margin": sink_zero_margin,
        "val_accuracy": m_val["accuracy"], "val_macro_f1": m_val["macro_f1"],
        "test_accuracy": m_test["accuracy"], "test_macro_f1": m_test["macro_f1"],
        "fast_predict_mode": "f64_gemm",
        "fast_predict_validated": gate["validated"],
        "fast_predict_n_checked": gate["n_checked"],
        "fast_predict_mismatches": gate["mismatches"],
        "fast_predict_min_abs_dec": gate["min_abs_dec"],
        "predict_ms_per_clip": float(np.mean(predict_ms)),
    }, drop_keys=keys)

    print(f"OK {variant}/seed{seed}  test {m_test['accuracy']:.4f}  "
          f"{np.mean(predict_ms):.2f} ms/clip", flush=True)
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
