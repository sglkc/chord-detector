"""Export Type-42 ESM figures for sn-supplementary-draft-5.tex.

Reads cnn-latest per-seed CSVs (orig/clean, seeds 42-46). Dataset-level
figures (chars, CQT shift, envelope, overlay CQT) stay in export_esm_figures.py.
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
RES = REPO / "training" / "notebooks" / "cnn-latest" / "results"
HIST = RES / "history"

OOD = ["thinkpad", "vivo", "flow", "thinkpad-2", "flow-2"]
COLOR = {"orig": "#1f4e79", "clean": "#c8791e"}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ARTICLE / f"{name}.pdf")
    fig.savefig(ARTICLE / f"{name}.png", dpi=140)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Validation curves of the ten reported 80/10/10 checkpoints
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.5), layout="constrained")
for v in ["orig", "clean"]:
    for s in range(42, 47):
        h = pd.read_csv(HIST / f"{v}-seed{s}.csv")
        hi = v == "clean" and s == 44  # worst val-loss, best mean OOD
        style = dict(
            color=COLOR[v],
            linewidth=1.4 if hi else 0.7,
            alpha=1.0 if hi else 0.5,
            linestyle="-",
        )
        axes[0].plot(h.epoch, h.val_accuracy, **style)
        axes[1].plot(h.epoch, h.val_loss, **style)
        if hi:
            axes[1].annotate(
                "clean/44\nrestored ep. 3",
                xy=(3, float(h.loc[h.epoch == 3, "val_loss"].iloc[0])),
                xytext=(18, 8e-3),
                fontsize=7,
                arrowprops=dict(arrowstyle="->", linewidth=0.6, color="0.35"),
            )
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
# Seed spread, CNN only: ten checkpoints, no earlier single-run overlay
# ---------------------------------------------------------------------------
ood = pd.read_csv(RES / "ood_seeds.csv")
fig, ax = plt.subplots(figsize=(6.85, 2.9), layout="constrained")
x = np.arange(len(OOD))
jitter = {"orig": -0.12, "clean": 0.12}
rng = np.random.default_rng(0)

for v in ["orig", "clean"]:
    sub = ood[ood.variant == v]
    for i, ds in enumerate(OOD):
        pts = sub[sub.dataset == ds]["accuracy"].to_numpy()
        xs = np.full(len(pts), x[i] + jitter[v]) + rng.uniform(-0.03, 0.03, len(pts))
        ax.scatter(
            xs, pts, s=22, facecolor="white", edgecolor=COLOR[v],
            linewidth=0.8, zorder=3, label=v if i == 0 else None,
        )
    means = sub.groupby("dataset")["accuracy"].mean().loc[OOD]
    ax.scatter(
        x + jitter[v], means, marker="_", s=170, color=COLOR[v],
        linewidth=1.4, zorder=4,
    )

ax.set_xticks(x, OOD)
ax.set_ylabel("Accuracy")
ax.set_ylim(0.62, 1.03)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2,
           columnspacing=1.2, handletextpad=0.4)
save(fig, "esm-seed-spread-cnn")


# ---------------------------------------------------------------------------
# Overlay tax per checkpoint: recorded accuracy minus overlay accuracy
# ---------------------------------------------------------------------------
ov = pd.read_csv(RES / "overlay_seeds.csv")
rec = ood.pivot_table(index=["variant", "seed"], columns="dataset", values="accuracy")
OVL = [("noise", "ESC-50 noise"), ("rir", "Room IR"), ("dir", "Microphone IR")]

fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.5), layout="constrained", sharey=True)
for ax, (key, title) in zip(axes, OVL):
    tab = ov[ov.overlay == key].pivot_table(
        index=["variant", "seed"], columns="dataset", values="accuracy",
    )
    tax = (rec - tab) * 100
    for j, v in enumerate(["orig", "clean"]):
        vals = tax.loc[v][OOD]
        off = -0.16 if j == 0 else 0.16
        for i, ds in enumerate(OOD):
            xs = np.full(5, i + off)
            ax.scatter(
                xs, vals[ds], s=18, facecolor="white", edgecolor=COLOR[v],
                linewidth=0.8, zorder=3, label=v if i == 0 else None,
            )
        ax.scatter(
            np.arange(len(OOD)) + off, vals.mean(),
            marker="_", s=120, color=COLOR[v], linewidth=1.3, zorder=4,
        )
    ax.axhline(0, color="0.4", linewidth=0.6)
    ax.set_xticks(np.arange(len(OOD)), OOD, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
axes[0].set_ylabel("Overlay tax (accuracy points)")
for ax, letter in zip(axes, "ABC"):
    ax.text(0.0, 1.06, f"({letter})", transform=ax.transAxes,
            fontweight="bold", fontsize=8)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
save(fig, "esm-overlay-tax")


# ---------------------------------------------------------------------------
# In-domain Techno-test overlay tax
# ---------------------------------------------------------------------------
ind = pd.read_csv(RES / "indomain_overlay_seeds.csv")
ind = ind[ind.overlay != "none"].copy()
ind["tax_pt"] = ind["tax"] * 100
order = ["noise", "rir", "dir"]
labels = {"noise": "ESC-50 noise", "rir": "Room IR", "dir": "Microphone IR"}

fig, ax = plt.subplots(figsize=(6.85, 2.4), layout="constrained")
for j, v in enumerate(["orig", "clean"]):
    sub = ind[ind.variant == v]
    off = -0.14 if j == 0 else 0.14
    for i, key in enumerate(order):
        pts = sub[sub.overlay == key]["tax_pt"].to_numpy()
        xs = np.full(len(pts), i + off) + rng.uniform(-0.03, 0.03, len(pts))
        ax.scatter(
            xs, pts, s=22, facecolor="white", edgecolor=COLOR[v],
            linewidth=0.8, zorder=3, label=v if i == 0 else None,
        )
        ax.scatter(
            [i + off], [pts.mean()], marker="_", s=160, color=COLOR[v],
            linewidth=1.4, zorder=4,
        )
ax.axhline(0, color="0.4", linewidth=0.6)
ax.set_xticks(np.arange(len(order)), [labels[k] for k in order])
ax.set_ylabel("Overlay tax (accuracy points)")
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2)
save(fig, "esm-indomain-overlay")


# ---------------------------------------------------------------------------
# 5-fold test accuracy (no per-epoch fold logs in cnn-latest)
# ---------------------------------------------------------------------------
kf = pd.read_csv(RES / "kfold.csv")
fig, ax = plt.subplots(figsize=(6.85, 2.35), layout="constrained")
w = 0.32
folds = np.arange(5)
for j, v in enumerate(["orig", "clean"]):
    acc = kf[kf.variant == v].sort_values("fold")["test_accuracy"].to_numpy()
    ax.bar(
        folds + (-w / 2 if j == 0 else w / 2), acc, width=w,
        facecolor="white", edgecolor=COLOR[v], linewidth=0.9, hatch="///" if j else "xxx",
        label=v,
    )
ax.set_xticks(folds, [f"Fold {i+1}" for i in folds])
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.9985, 1.0003)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
fig.legend(loc="outside lower center", ncol=2)
save(fig, "esm-cnn-kfold")
