"""Export Type-42 result figures for sn-article-draft-8.tex.

Reads the same cnn-latest per-seed CSVs the article tabulates (orig/clean,
seeds 42-46), so nothing here depends on the superseded cnn-v2 outputs.
Style follows .grok/skills/article-workflow/references/figures.md: outlined
marks for print, legends outside the axes, Springer full width 6.85 in.
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

TECHNO = ["thinkpad", "vivo", "flow"]
YAMAHA = ["thinkpad-2", "flow-2"]
COLOR = {"orig": "#1f4e79", "clean": "#c8791e"}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ARTICLE / f"{name}.pdf")
    fig.savefig(ARTICLE / f"{name}.png", dpi=160)
    plt.close(fig)
    print("wrote", name)


ood = pd.read_csv(RES / "ood_seeds.csv")
ov = pd.read_csv(RES / "overlay_seeds.csv")
ind = pd.read_csv(RES / "indomain_overlay_seeds.csv")

fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.85), layout="constrained")

# ---------------------------------------------------------------------------
# (A) which recorded factor is larger, one point per matched comparison
# ---------------------------------------------------------------------------
piv = ood.pivot_table(index=["variant", "seed"], columns="dataset", values="accuracy")
rows = []
for (v, s), r in piv.iterrows():
    for te, ya in [("thinkpad", "ThinkPad"), ("flow", "Flow")]:
        pair_ya = "thinkpad-2" if te == "thinkpad" else "flow-2"
        rows.append((v, ya, (1 - r[te]) * 100, (r[te] - r[pair_ya]) * 100))
fac = pd.DataFrame(rows, columns=["variant", "pair", "rs", "kb"])

ax = axes[0]
lim = 26
ax.plot([0, lim], [0, lim], color="0.35", linewidth=0.8, linestyle=(0, (4, 3)),
        zorder=1)
ax.text(19.0, 19.6, "equal cost", fontsize=6.5, color="0.35", rotation=45,
        ha="center", va="bottom", rotation_mode="anchor")
MARK = {("orig", "ThinkPad"): "o", ("orig", "Flow"): "^",
        ("clean", "ThinkPad"): "s", ("clean", "Flow"): "D"}
handles_a = []
for (v, pair), mk in MARK.items():
    sub = fac[(fac.variant == v) & (fac.pair == pair)]
    h = ax.scatter(sub.rs, sub.kb, s=30, marker=mk, facecolor="white",
                   edgecolor=COLOR[v], linewidth=1.0, zorder=3,
                   label=f"{v}, {pair} pair")
    handles_a.append(h)
ax.set_xlim(-1.2, lim)
ax.set_ylim(-2.5, lim)
ax.set_xlabel("Recording-setting drop (accuracy points)")
ax.set_ylabel("Keyboard gap (accuracy points)")
ax.set_axisbelow(True)
ax.grid(True, linestyle=":", linewidth=0.4, color="0.8")

# ---------------------------------------------------------------------------
# (B) what each overlay removes, mean over checkpoints with min-max range
# ---------------------------------------------------------------------------
base = ood.set_index(["variant", "seed", "dataset"])["accuracy"]
ov = ov.copy()
ov["drop"] = [(base[(r.variant, r.seed, r.dataset)] - r.accuracy) * 100
              for r in ov.itertuples()]
ind = ind[ind.overlay != "none"].copy()
ind["drop"] = ind["tax"] * 100

OVL = [("noise", "ESC-50\nnoise"), ("rir", "Room\nIR"), ("dir", "Microphone\nIR")]
GROUPS = [("Techno test split", "0.15", ""),
          ("Techno, recorded shift", COLOR["orig"], "///"),
          ("Yamaha, recorded shift", COLOR["clean"], "xxx")]

ax = axes[1]
w = 0.26
handles_b = []
for gi, (glab, gcol, hatch) in enumerate(GROUPS):
    means, lo, hi = [], [], []
    for key, _ in OVL:
        if gi == 0:
            vals = ind[ind.overlay == key]["drop"].to_numpy()
        else:
            pool = TECHNO if gi == 1 else YAMAHA
            vals = ov[(ov.overlay == key) & (ov.dataset.isin(pool))]["drop"].to_numpy()
        means.append(vals.mean())
        lo.append(vals.mean() - vals.min())
        hi.append(vals.max() - vals.mean())
    x = np.arange(len(OVL)) + (gi - 1) * w
    b = ax.bar(x, means, width=w, facecolor="white", edgecolor=gcol,
               linewidth=0.9, hatch=hatch, zorder=3, label=glab)
    ax.errorbar(x, means, yerr=[lo, hi], fmt="none", ecolor=gcol,
                elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
    handles_b.append(b)

ax.axhline(0, color="0.35", linewidth=0.6)
ax.set_xticks(np.arange(len(OVL)), [lab for _, lab in OVL])
ax.set_ylabel("Accuracy removed (points)")
ax.set_ylim(-2, 30)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.8")

# --- panel letters above the axes, legends outside them --------------------
for ax_, letter in zip(axes, "AB"):
    ax_.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=8, pad=4)

leg_a = fig.legend(handles=handles_a, loc="outside lower left", ncol=2,
                   frameon=False, columnspacing=1.0, handletextpad=0.35,
                   labelspacing=0.3)
fig.legend(handles=handles_b, loc="outside lower right", ncol=1, frameon=False,
           handletextpad=0.5, labelspacing=0.3, handlelength=1.6)
save(fig, "fig-factors-overlays")
