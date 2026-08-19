"""Export Type-42 ESM figures for the per-seed protocol (sn-supplementary-draft-3.tex).

Companion to export_esm_figures.py, which draws the dataset- and single-checkpoint
figures. Everything here reads the 15-checkpoint cross-eval outputs and the
nearest-centroid baseline, so no figure below depends on the originally published
n=1 model.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": False,
})

REPO = Path("/home/seya/code/chord-detection")
ARTICLE = REPO / "article"
NB = REPO / "training" / "notebooks"
CE = NB / "cnn-v2" / "cross-eval-results"
BASE = NB / "baseline-centroid" / "baseline-results"

OOD = ["thinkpad", "vivo", "flow", "thinkpad-2", "flow-2"]
VARIANTS = ["orig", "clean", "insnorm"]
COLOR = {"orig": "#1f4e79", "clean": "#c8791e", "insnorm": "#1b6b45"}
PUBLISHED = {"thinkpad": 0.994, "vivo": 0.946, "flow": 0.944,
             "thinkpad-2": 0.885, "flow-2": 0.874}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ARTICLE / f"{name}.pdf")
    fig.savefig(ARTICLE / f"{name}.png", dpi=140)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Seed spread: every checkpoint on every recorded held-out dataset, against the
# zero-parameter centroid baseline and the originally published single model
# ---------------------------------------------------------------------------
ood = pd.read_csv(CE / "pooled_ood_seeds.csv")
baseline = pd.read_csv(BASE / "summary.csv")
base_flag = baseline[(baseline.config == "clean") & (baseline.metric == "corr")] \
    .set_index("dataset")["accuracy"]

fig, ax = plt.subplots(figsize=(6.85, 2.9), layout="constrained")
x = np.arange(len(OOD))
jitter = {"orig": -0.22, "clean": 0.0, "insnorm": 0.22}
rng = np.random.default_rng(0)

for v in VARIANTS:
    sub = ood[ood.variant == v]
    for i, ds in enumerate(OOD):
        rows = sub[sub.dataset == ds]
        collapse = ((rows.variant == "insnorm") & (rows.seed == 43)).to_numpy()
        pts = rows["accuracy"].to_numpy()
        xs = np.full(len(pts), x[i] + jitter[v]) + rng.uniform(-0.03, 0.03, len(pts))
        ax.scatter(xs[~collapse], pts[~collapse], s=22, facecolor=COLOR[v],
                   edgecolor=COLOR[v], linewidth=0.7, zorder=3,
                   label=v if i == 0 else None)
        ax.scatter(xs[collapse], pts[collapse], s=22, facecolor="white",
                   edgecolor=COLOR[v], linewidth=0.7, zorder=3)
    keep = sub[~((sub.variant == "insnorm") & (sub.seed == 43))]
    means = keep.groupby("dataset")["accuracy"].mean().loc[OOD]
    ax.scatter(x + jitter[v], means, marker="_", s=170, color=COLOR[v],
               linewidth=1.4, zorder=4)

ax.plot(x, base_flag.loc[OOD], linestyle="--", linewidth=0.9, color="black",
        marker="D", markersize=3.4, zorder=5, label="centroid baseline")
ax.scatter(x, [PUBLISHED[d] for d in OOD], marker="x", s=30, color="0.35",
           linewidth=1.0, zorder=5, label="published model ($n{=}1$)")

ax.set_xticks(x, OOD)
ax.set_ylabel("Accuracy")
ax.set_ylim(-0.03, 1.05)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=5,
           columnspacing=1.2, handletextpad=0.4)
save(fig, "esm-seed-spread")


# ---------------------------------------------------------------------------
# Overlay tax per checkpoint: recorded accuracy minus overlay accuracy
# ---------------------------------------------------------------------------
ov = pd.read_csv(CE / "overlay_pooled_seeds.csv")
rec = ood.pivot_table(index=["variant", "seed"], columns="dataset", values="accuracy")
OVL = [("noise", "ESC-50 noise"), ("rir", "Room IR"), ("dir", "Microphone IR")]

fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.5), layout="constrained", sharey=True)
for ax, (key, title) in zip(axes, OVL):
    tab = ov[ov.overlay == key].pivot_table(index=["variant", "seed"],
                                            columns="dataset", values="accuracy")
    tax = (rec - tab) * 100
    tax = tax.drop(index=("insnorm", 43), errors="ignore")
    for j, v in enumerate(["orig", "clean"]):
        vals = tax.loc[v][OOD]
        for i, ds in enumerate(OOD):
            xs = np.full(5, i + (-0.16 if j == 0 else 0.16))
            ax.scatter(xs, vals[ds], s=16, facecolor=COLOR[v], edgecolor="none",
                       alpha=0.85, zorder=3, label=v if i == 0 else None)
        ax.scatter(np.arange(len(OOD)) + (-0.16 if j == 0 else 0.16),
                   vals.mean(), marker="_", s=120, color=COLOR[v],
                   linewidth=1.3, zorder=4)
    ax.axhline(0, color="0.4", linewidth=0.6)
    ax.set_xticks(np.arange(len(OOD)), OOD, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
axes[0].set_ylabel("Overlay tax (accuracy points)")
axes[0].legend(loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.3)
for ax, letter in zip(axes, "ABC"):
    ax.text(0.0, 1.06, f"({letter})", transform=ax.transAxes,
            fontweight="bold", fontsize=8)
save(fig, "esm-overlay-tax")


# ---------------------------------------------------------------------------
# Centroid baseline: which normalization step buys the out-of-domain accuracy
# ---------------------------------------------------------------------------
METRICS = [("euclid", "Raw distance"), ("scale_only", "Scale only"),
           ("center_only", "Centering only"), ("cos_nocenter", "Cosine, no centering"),
           ("corr", "Centering $+$ scale")]
summ = baseline[baseline.config == "clean"].pivot_table(
    index="metric", columns="dataset", values="accuracy")
ovb = pd.read_csv(BASE / "overlay_summary.csv")
ovb = ovb[ovb.config == "clean"]

fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.6), layout="constrained", sharey=True)

ax = axes[0]
w = 0.16
for k, (m, lab) in enumerate(METRICS):
    ax.bar(np.arange(len(OOD)) + (k - 2) * w, summ.loc[m, OOD], width=w,
           label=lab, edgecolor="0.25", linewidth=0.4)
ax.set_xticks(np.arange(len(OOD)), OOD, rotation=35, ha="right")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.4, 1.02)
ax.set_title("Recorded shift")
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")

ax = axes[1]
agg = ovb.pivot_table(index="metric", columns="overlay", values="accuracy")
order = ["noise", "rir", "dir"]
labels = {"noise": "Noise", "rir": "Room IR", "dir": "Mic IR"}
for k, (m, lab) in enumerate(METRICS):
    ax.bar(np.arange(len(order)) + (k - 2) * w, agg.loc[m, order], width=w,
           label=lab, edgecolor="0.25", linewidth=0.4)
ax.set_xticks(np.arange(len(order)), [labels[o] for o in order])
ax.set_title("Eval-only overlays (mean over datasets)")
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")

for ax, letter in zip(axes, "AB"):
    ax.text(0.0, 1.06, f"({letter})", transform=ax.transAxes,
            fontweight="bold", fontsize=8)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5,
           columnspacing=1.0, handletextpad=0.4)
save(fig, "esm-baseline-metrics")


# ---------------------------------------------------------------------------
# Per-checkpoint training curves, all ten orig/clean runs on one pair of axes
# ---------------------------------------------------------------------------
HIST = NB / "cnn-v2" / "variant-results" / "history"
fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.5), layout="constrained")
for v in ["orig", "clean"]:
    for s in [42, 43, 44, 45, 46]:
        h = pd.read_csv(HIST / f"{v}-seed{s}.csv")
        hi = v == "clean" and s == 42
        style = dict(color=COLOR[v], linewidth=1.4 if hi else 0.7,
                     alpha=1.0 if hi else 0.45,
                     linestyle="-" if not hi else "-")
        axes[0].plot(h.epoch, h.val_accuracy, **style)
        axes[1].plot(h.epoch, h.val_loss, **style)
        if hi:
            axes[0].annotate("clean/42\nrestored ep. 4", xy=(4, h.val_accuracy[3]),
                             xytext=(6.5, 0.55), fontsize=7,
                             arrowprops=dict(arrowstyle="->", linewidth=0.6,
                                             color="0.35"))
axes[0].set_ylabel("Validation accuracy")
axes[0].set_ylim(0.0, 1.03)
axes[1].set_ylabel("Validation loss")
axes[1].set_yscale("log")
for ax, letter in zip(axes, "AB"):
    ax.set_xlabel("Epoch")
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", linewidth=0.4, color="0.75")
    ax.text(0.0, 1.06, f"({letter})", transform=ax.transAxes,
            fontweight="bold", fontsize=8)
handles = [plt.Line2D([], [], color=COLOR[v], linewidth=1.2, label=v)
           for v in ["orig", "clean"]]
fig.legend(handles=handles, loc="outside lower center", ncol=2)
save(fig, "esm-variant-history")


# ---------------------------------------------------------------------------
# Seed spread, CNN only: the ten reported checkpoints, no third variant and no
# centroid baseline (sn-supplementary-draft-4.tex)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.85, 2.9), layout="constrained")
jitter2 = {"orig": -0.12, "clean": 0.12}
rng = np.random.default_rng(0)

for v in ["orig", "clean"]:
    sub = ood[ood.variant == v]
    for i, ds in enumerate(OOD):
        pts = sub[sub.dataset == ds]["accuracy"].to_numpy()
        xs = np.full(len(pts), x[i] + jitter2[v]) + rng.uniform(-0.03, 0.03, len(pts))
        ax.scatter(xs, pts, s=22, facecolor=COLOR[v], edgecolor=COLOR[v],
                   linewidth=0.7, zorder=3, label=v if i == 0 else None)
    means = sub.groupby("dataset")["accuracy"].mean().loc[OOD]
    ax.scatter(x + jitter2[v], means, marker="_", s=170, color=COLOR[v],
               linewidth=1.4, zorder=4)

ax.scatter(x, [PUBLISHED[d] for d in OOD], marker="x", s=30, color="0.35",
           linewidth=1.0, zorder=5, label="earlier single checkpoint")

ax.set_xticks(x, OOD)
ax.set_ylabel("Accuracy")
ax.set_ylim(0.25, 1.03)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3,
           columnspacing=1.2, handletextpad=0.4)
save(fig, "esm-seed-spread-cnn")
