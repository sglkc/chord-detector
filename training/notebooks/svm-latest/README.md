# svm-latest

RBF-SVM baseline re-run under the current protocol, replacing the single `n=1` fit on the
old dataset pool that thesis §4.6 and §4.7 still report.

## Why this exists

Two things were wrong with the old baseline.

**Protocol.** The SVM numbers came from one fit on a superseded pool while the CNN now has
ten checkpoints on the 7,200 / 7,172 pools. `RENCANA-SINKRONISASI-ARTIKEL.md:265` flags
both subsections as TODO for exactly this reason. The §4.7 comparison table reads `1,0000`
against `1,0000` and concludes the models cannot be told apart — an artifact of comparing
only in domain, where both saturate.

**A sink class.** Out of domain the old SVM funnels predictions into `C_diminished_4`:
37% of `thinkpad-2`, 17% of `flow-2`, 15% of `thinkpad`, against a true rate of 2.8%. On an
older noisy evaluation it scored exactly `40/1440` — every clip predicted `C_diminished_4`.

`C_diminished_4` is also the class holding the 16 digital-silence clips, which in
`training.npz` are exactly `0.0` in all 40,608 dimensions while real clips average about
−55 dB. The initial hypothesis was that those clips *caused* the sink.

**The ablation refuted that.** Removing them does remove the training-set outlier
(`train_l2_max` falls from ~1237 to 674), but the sink stays `C_diminished_4` and gets
*larger*, and out-of-domain accuracy gets *worse*. The real mechanism, which the same
diagnostics measure directly:

- Scaled row norms inflate out of domain: median 190–198 in domain against 270–397 on the
  recorded sets. Mean max kernel value falls from 0.76–0.79 in domain to 0.36–0.55.
- As every kernel value collapses toward zero, each one-vs-one decision reduces to its
  intercept, so the vote vector becomes constant and one class absorbs everything. That
  class is computable from `intercept_` signs alone, before any data is seen
  (`sink_class_zero_kernel`), and it matches the observed sink on every out-of-domain set
  while matching on none of the in-domain ones. Its winning margin is **one vote**, so
  *which* class becomes the sink is close to arbitrary.
- Removing the silent clips made this worse because those 16 rows were inflating the
  scaler's per-dimension variance. A tighter scaler produces larger scaled norms, which
  pushes out-of-domain rows further outside the kernel's support.

So the SVM's out-of-domain failure is RBF support collapse under distribution shift, not
a data defect. The silent clips were incidentally *damping* it.

The CNN does not collapse this way. Its most-predicted class on `thinkpad-2` takes
5.3–10.4%, and it is a *different* class on each of the ten checkpoints: ordinary
confusion, no attractor. Its first layer is `BatchNormalization(axis=1)`, which
renormalizes per frequency bin at inference time, so a global gain or tilt shift does not
move it off-manifold the way a fixed `StandardScaler` plus a global distance metric does.

One correction the numbers force: **the SVM is not uniformly worse.** It ties the CNN in
domain and beat it on `flow` (0.994 vs 0.953) and `vivo` (0.978 vs 0.915) on the old fit.
The deficit is concentrated on `thinkpad` and the keyboard shift.

## Protocol

Partitions are **not** redrawn. Every run indexes its pool by the `source_index` column of
`cnn-latest/results/splits/{variant}-{tag}.csv`, so each SVM checkpoint is paired with a
CNN checkpoint rather than merely matched in protocol. `SVC` is deterministic given its
data, so the split is the only source of run-to-run variation — "five seeds" means five
splits.

| variant | pool | partition |
| --- | --- | --- |
| `orig` | `features/training.npz` (7,200) | `orig-seed{s}` |
| `clean` | `features/training-clean.npz` (7,172) | `clean-seed{s}` |
| `clean-fixed` | `training.npz` minus the 28 rows in `excluded.csv` | `orig-seed{s}`, rows dropped in place |

`clean` changes the data *and* redraws the split; `clean-fixed` changes only the data.
So `clean-fixed − orig` isolates the 28 clips and `clean − clean-fixed` isolates the redraw.
This is the run `sn-article-draft-8.tex:223` concedes is missing for the CNN — for the SVM
it is cheap.

`C=1000, gamma=1e-5` are frozen everywhere, inherited from the published tuning, so the
comparison is an ablation and not a re-tuning. Fitting uses the `train` split only, never
`train+val`, so the SVM sees exactly the rows the CNN trained on. That means these runs
will **not** reproduce `models/svm-best`, which was fit on a 90/10 split of the whole pool.

## Fast prediction

`SVC.predict` is single-threaded libsvm streaming a 1.35 GB float64 support-vector array
once per clip: 0.19–0.43 s/clip measured, memory-bandwidth bound. The full campaign would
be ~20 hours of prediction alone.

`fast_rbf.FastRBF` recomputes the identical decision values with one GEMM per chunk plus 36
small per-class matmuls, at ~2 ms/clip — **~95× faster**, campaign time in minutes.

It is float64 by default, and that was a measurement rather than a preference. Against
`svm-best`, float32 gives decision values accurate to 2.4e-7 against a smallest observed
|decision| of 8.1e-6 — a 34× margin, which will not survive the ~7.5e7 decision values a
full campaign evaluates. float64 costs 1.7e-14, a margin near 5e8, for 2.0 ms/clip against
1.0. Since even the slow path is ~95× libsvm there is nothing to buy by approximating.

Separately, the squared-norm terms are accumulated in float64 *regardless* of input dtype.
Summing 40,608 squares in float32 loses ~1e-2 on a norm near 1e5 and injects ~1.8e-6 of
decision error — worse than the GEMM precision it was meant to save on.

Every fitted model is gated: `validate_against_sklearn` asserts exact label agreement with
`SVC.predict` on 128 rows drawn half from the model's own test split and half from `vivo`
(the highest-norm, lowest-kernel-value set, where sign flips are most likely). It raises
rather than falling back silently.

## Files

| file | purpose |
| --- | --- |
| `fast_rbf.py` | `FastRBF` + the validation gate |
| `svm_common.py` | variant/split loading, streaming scaler, metrics, sink diagnostics |
| `gate_svmbest.py` | step zero — proves the reconstruction against the published model |
| `run_one.py` | fit + evaluate one `(variant, seed)`, isolated process |
| `run_kfold.py` | one `(variant, fold)` of the in-domain 5-fold |
| `train-variants.ipynb` | driver for the seed runs |
| `kfold.ipynb` | driver for the folds |
| `compare.ipynb` | CNN vs SVM tables and the sink-class ablation |

Fitting and evaluation share one process, unlike `cnn-latest`. A Keras checkpoint is
~124 MB; a fitted `SVC` here is ~1.35 GB, so persisting 15 of them to reload later costs
20 GB and minutes per load. The process boundary is still per-run — process exit is the
only thing that reliably returns that memory to the OS. Use `--save-bundle` to keep a
model, which writes a ~673 MB float32 `.npz` loadable straight into `FastRBF`.

`results/` mirrors `cnn-latest/results/` column for column (`progress.csv`, `kfold.csv`,
`ood_seeds.csv`, `per_class_recall_seeds.csv`, `confusion/`), so a comparison table is a
`pd.concat`. SVM-specific quantities live in sibling files: `diagnostics.csv`,
`support_seeds.csv`, `pred_hist_seeds.csv`, `sink_seeds.csv`,
`quality_root_recall_seeds.csv`.

Eval-only overlays (noise / room IR / mic IR) are scored by `run_overlay.py` into
`results/overlay_seeds.csv`, same columns as `cnn-latest`.
