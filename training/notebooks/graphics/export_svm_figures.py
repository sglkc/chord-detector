#!/usr/bin/env python3
"""Export Type-42 figures for the svm-latest baseline experiment.

Reads only the generated CSVs under training/notebooks/svm-latest/results and
training/notebooks/cnn-latest/results, so nothing here depends on a live notebook.

Style follows .grok/skills/article-workflow/references/figures.md: outlined marks for
print, hatch when more than two series share a hue, legends outside the axes via
fig.legend(loc='outside ...') with layout='constrained', Springer full width 6.85 in,
8 pt labels, Type-42 PDF plus a 160 dpi PNG preview.

Writes article/svm-{ablation,collapse,seed-spread,mechanism}.{pdf,png}.
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
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
})

REPO = Path("/home/seya/code/chord-detection")
ARTICLE = REPO / "article"
SVM = REPO / "training" / "notebooks" / "svm-latest" / "results"
CNN = REPO / "training" / "notebooks" / "cnn-latest" / "results"

DATASETS = ["thinkpad", "vivo", "flow", "thinkpad-2", "flow-2"]
DS_LABEL = {"thinkpad": "ThinkPad", "vivo": "Vivo", "flow": "Flow",
            "thinkpad-2": "ThinkPad-2\n(Yamaha)", "flow-2": "Flow-2\n(Yamaha)"}
VARIANTS = ["orig", "clean-fixed", "clean"]
V_LABEL = {"orig": "orig", "clean-fixed": "clean-fixed", "clean": "clean"}

INK = "#1f2933"
CNN_C = "#1f4e79"
SVM_C = "#c8791e"
HATCH = {"orig": "", "clean-fixed": "///", "clean": "xxx"}


def save(fig, name):
    fig.savefig(ARTICLE / f"{name}.pdf")
    fig.savefig(ARTICLE / f"{name}.png", dpi=160)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- load
svm_ood = pd.read_csv(SVM / "ood_seeds.csv")
cnn_ood = pd.read_csv(CNN / "ood_seeds.csv")
sink = pd.read_csv(SVM / "sink_seeds.csv")
diag = pd.read_csv(SVM / "diagnostics.csv")

# CNN sink share, recomputed from its confusion matrices
rows = []
for f in sorted((CNN / "confusion").glob("*.csv")):
    ds, variant, seed = f.stem.rsplit("_", 2)
    col = pd.read_csv(f, index_col=0).sum(0)
    rows.append({"dataset": ds, "variant": variant, "seed": seed,
                 "sink_class": col.idxmax(),
                 "excess": (col.max() / col.sum() - 1 / 36) * 100})
cnn_sink = pd.DataFrame(rows)


# ============================================================ FIGURE 1
# The ablation moves the two models in opposite directions.
fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.9), layout="constrained")

ax = axes[0]
sm = svm_ood.groupby("variant")["accuracy"].mean() * 100
cm = cnn_ood.groupby("variant")["accuracy"].mean() * 100
xpos = {"orig": 0, "clean-fixed": 1, "clean": 2}
h = []
h.append(ax.plot([0, 2], [cm["orig"], cm["clean"]], color=CNN_C, linewidth=1.2,
                 marker="o", markersize=6, markerfacecolor="white",
                 markeredgewidth=1.2, label="CNN")[0])
h.append(ax.plot([xpos[v] for v in VARIANTS], [sm[v] for v in VARIANTS],
                 color=SVM_C, linewidth=1.2, marker="s", markersize=6,
                 markerfacecolor="white", markeredgewidth=1.2, label="SVM")[0])
ax.annotate(f"{cm['clean'] - cm['orig']:+.1f} pts", (2, cm["clean"]),
            textcoords="offset points", xytext=(-4, 7), ha="right",
            fontsize=7, color=CNN_C)
ax.annotate(f"{sm['clean'] - sm['orig']:+.1f} pts", (2, sm["clean"]),
            textcoords="offset points", xytext=(-4, 8), ha="right",
            fontsize=7, color=SVM_C)
ax.set_xticks([0, 1, 2], [V_LABEL[v] for v in VARIANTS])
ax.set_xlim(-0.35, 2.5)
ax.set_ylabel("Mean accuracy over 5 recorded sets (%)")
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.8")
ax.set_title("(A)", loc="left", fontweight="bold", fontsize=8, pad=4)

ax = axes[1]
d_svm = (svm_ood[svm_ood.variant == "clean"].groupby("dataset")["accuracy"].mean()
         - svm_ood[svm_ood.variant == "orig"].groupby("dataset")["accuracy"].mean()) * 100
d_cnn = (cnn_ood[cnn_ood.variant == "clean"].groupby("dataset")["accuracy"].mean()
         - cnn_ood[cnn_ood.variant == "orig"].groupby("dataset")["accuracy"].mean()) * 100
x = np.arange(len(DATASETS))
w = 0.36
b1 = ax.bar(x - w / 2, [d_cnn[d] for d in DATASETS], width=w, facecolor="white",
            edgecolor=CNN_C, linewidth=0.9, zorder=3, label="CNN")
b2 = ax.bar(x + w / 2, [d_svm[d] for d in DATASETS], width=w, facecolor="white",
            edgecolor=SVM_C, linewidth=0.9, hatch="///", zorder=3, label="SVM")
ax.axhline(0, color=INK, linewidth=0.7)
ax.set_xticks(x, [DS_LABEL[d] for d in DATASETS])
ax.set_ylabel("Accuracy change, clean − orig (points)")
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.8")
ax.set_title("(B)", loc="left", fontweight="bold", fontsize=8, pad=4)

fig.legend(handles=[h[0], h[1]], loc="outside lower center", ncol=2,
           frameon=False, columnspacing=2.0)
save(fig, "svm-ablation")


# ============================================================ FIGURE 2
# The collapse: one class absorbs predictions, and only for the SVM.
fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.9), layout="constrained")

ax = axes[0]
x = np.arange(len(DATASETS))
w = 0.2
handles = []
series = [("CNN orig", cnn_sink[cnn_sink.variant == "orig"].groupby("dataset")["excess"].mean(),
           CNN_C, ""),
          ("SVM orig", sink[sink.variant == "orig"].groupby("dataset")["sink_share_excess"].mean() * 100,
           SVM_C, ""),
          ("SVM clean-fixed", sink[sink.variant == "clean-fixed"].groupby("dataset")["sink_share_excess"].mean() * 100,
           SVM_C, "///"),
          ("SVM clean", sink[sink.variant == "clean"].groupby("dataset")["sink_share_excess"].mean() * 100,
           SVM_C, "xxx")]
for i, (lab, vals, col, hh) in enumerate(series):
    b = ax.bar(x + (i - 1.5) * w, [vals.get(d, np.nan) for d in DATASETS], width=w,
               facecolor="white", edgecolor=col, linewidth=0.8, hatch=hh,
               zorder=3, label=lab)
    handles.append(b)
ax.axhline(0, color=INK, linewidth=0.7)
ax.set_xticks(x, [DS_LABEL[d] for d in DATASETS])
ax.set_ylabel("Excess share of one class (points over 1/36)")
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.8")
ax.set_title("(A)", loc="left", fontweight="bold", fontsize=8, pad=4)

ax = axes[1]
n_svm = sink[sink.dataset.isin(DATASETS)].groupby(
    ["variant", "dataset"])["sink_class"].nunique().groupby("dataset").max()
n_cnn = cnn_sink.groupby(["variant", "dataset"])["sink_class"].nunique().groupby("dataset").max()
w = 0.36
ax.bar(x - w / 2, [n_cnn[d] for d in DATASETS], width=w, facecolor="white",
       edgecolor=CNN_C, linewidth=0.9, zorder=3)
ax.bar(x + w / 2, [n_svm[d] for d in DATASETS], width=w, facecolor="white",
       edgecolor=SVM_C, linewidth=0.9, hatch="///", zorder=3)
ax.set_xticks(x, [DS_LABEL[d] for d in DATASETS])
ax.set_yticks(range(0, 6))
ax.set_ylabel("Distinct most-predicted classes\nacross 5 checkpoints")
ax.set_ylim(0, 5.8)
ax.axhline(1, color="0.45", linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
ax.text(4.45, 1.12, "one fixed attractor", fontsize=6.5, color="0.35", ha="right")
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.8")
ax.set_title("(B)", loc="left", fontweight="bold", fontsize=8, pad=4)

fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False,
           columnspacing=1.6, handletextpad=0.4)
save(fig, "svm-collapse")


# ============================================================ FIGURE 3
# Mechanism: cleaning tightens the scaler, inflating norms and killing kernel support.
fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.9), layout="constrained")

ax = axes[0]
agg = sink.groupby(["variant", "dataset"])[["scaled_l2_median", "k_max_mean"]].mean()
MARK = {"thinkpad": "o", "vivo": "^", "flow": "D", "thinkpad-2": "s",
        "flow-2": "v", "test": "P", "val": "X"}
ds_handles = []
for ds in ["test"] + DATASETS:
    for v, col in (("orig", INK), ("clean", SVM_C)):
        if (v, ds) not in agg.index:
            continue
        r = agg.loc[(v, ds)]
        h = ax.scatter(r.scaled_l2_median, r.k_max_mean, s=34, marker=MARK[ds],
                       facecolor="white", edgecolor=col, linewidth=1.0, zorder=3,
                       label=("Techno test split" if ds == "test" else
                              DS_LABEL[ds].replace("\n", " ")))
        if v == "orig":
            ds_handles.append(h)
    if ("orig", ds) in agg.index and ("clean", ds) in agg.index:
        a, b = agg.loc[("orig", ds)], agg.loc[("clean", ds)]
        ax.annotate("", xy=(b.scaled_l2_median, b.k_max_mean),
                    xytext=(a.scaled_l2_median, a.k_max_mean),
                    arrowprops=dict(arrowstyle="->", color="0.55", lw=0.7,
                                    shrinkA=4, shrinkB=4), zorder=2)
ax.set_xlabel("Median scaled row norm")
ax.set_ylabel("Mean max kernel value")
ax.set_axisbelow(True)
ax.grid(True, linestyle=":", linewidth=0.4, color="0.85")
ax.text(0.97, 0.93, "in domain", transform=ax.transAxes, ha="right",
        fontsize=6.5, color="0.35")
ax.text(0.06, 0.10, "out of domain", transform=ax.transAxes, ha="left",
        fontsize=6.5, color="0.35")
ax.set_title("(A)", loc="left", fontweight="bold", fontsize=8, pad=4)

ax = axes[1]
tl = diag.groupby("variant")[["train_l2_median", "train_l2_max"]].mean()
x2 = np.arange(len(VARIANTS))
w = 0.36
b_handles = [
    ax.bar(x2 - w / 2, [tl.loc[v, "train_l2_median"] for v in VARIANTS], width=w,
           facecolor="white", edgecolor=INK, linewidth=0.9, zorder=3,
           label="median training row"),
    ax.bar(x2 + w / 2, [tl.loc[v, "train_l2_max"] for v in VARIANTS], width=w,
           facecolor="white", edgecolor=SVM_C, linewidth=0.9, hatch="///", zorder=3,
           label="largest training row"),
]
ax.set_xticks(x2, [V_LABEL[v] for v in VARIANTS])
ax.set_ylabel("Scaled row norm")
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.8")
ax.annotate("16 silent clips\nremoved", (1, tl.loc["clean-fixed", "train_l2_max"]),
            textcoords="offset points", xytext=(6, 14), fontsize=6.5, color="0.35",
            arrowprops=dict(arrowstyle="->", color="0.55", lw=0.7))
ax.set_title("(B)", loc="left", fontweight="bold", fontsize=8, pad=4)

from matplotlib.lines import Line2D  # noqa: E402

variant_keys = [
    Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
           markeredgecolor=INK, markeredgewidth=1.0, markersize=5, label="orig"),
    Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
           markeredgecolor=SVM_C, markeredgewidth=1.0, markersize=5, label="clean"),
]
leg_a = fig.legend(handles=ds_handles + variant_keys, loc="outside lower left",
                   ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.35,
                   labelspacing=0.3, fontsize=7)
fig.legend(handles=b_handles, loc="outside lower right", ncol=1, frameon=False,
           columnspacing=1.4, labelspacing=0.3, handlelength=1.6)
save(fig, "svm-mechanism")


# ============================================================ FIGURE 4
# Every checkpoint, both models.
fig, ax = plt.subplots(figsize=(6.85, 3.0), layout="constrained")
groups = [("CNN", "orig", CNN_C, "o"), ("CNN", "clean", CNN_C, "s"),
          ("SVM", "orig", SVM_C, "o"), ("SVM", "clean-fixed", SVM_C, "^"),
          ("SVM", "clean", SVM_C, "s")]
x = np.arange(len(DATASETS))
span = 0.72
handles = []
for i, (model, v, col, mk) in enumerate(groups):
    src = cnn_ood if model == "CNN" else svm_ood
    off = (i - (len(groups) - 1) / 2) * (span / len(groups))
    xs, ys = [], []
    for j, ds in enumerate(DATASETS):
        vals = src[(src.variant == v) & (src.dataset == ds)]["accuracy"].to_numpy() * 100
        xs.extend([j + off] * len(vals))
        ys.extend(vals)
        if len(vals):
            ax.plot([j + off - 0.055, j + off + 0.055], [vals.mean()] * 2,
                    color=col, linewidth=1.4, zorder=4, solid_capstyle="butt")
    h = ax.scatter(xs, ys, s=15, marker=mk, facecolor="white", edgecolor=col,
                   linewidth=0.8, zorder=3, label=f"{model} {v}")
    handles.append(h)
ax.set_xticks(x, [DS_LABEL[d] for d in DATASETS])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(30, 102)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.8")
for j in range(len(DATASETS) - 1):
    ax.axvline(j + 0.5, color="0.9", linewidth=0.6, zorder=1)
fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
           columnspacing=1.4, handletextpad=0.35)
save(fig, "svm-seed-spread")
