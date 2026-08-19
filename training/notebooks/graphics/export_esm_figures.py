"""Export Type-42 ESM figures for sn-supplementary-draft.tex."""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

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
DATA = REPO / "training" / "datasets"
NB = REPO / "training" / "notebooks"
FEAT = REPO / "training" / "features"

DATASETS = ["training", "thinkpad", "vivo", "flow", "thinkpad-2", "flow-2"]
OOD = ["thinkpad", "vivo", "flow", "thinkpad-2", "flow-2"]
_QUAL = {"major": "maj", "minor": "min", "diminished": "dim"}


def pretty_class(name: str) -> str:
    body = str(name).removesuffix("_4")
    root, qual = body.rsplit("_", 1)
    return f"{root.replace('#', '♯')} {_QUAL.get(qual, qual)}"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ARTICLE / f"{name}.pdf")
    fig.savefig(ARTICLE / f"{name}.png", dpi=140)
    plt.close(fig)
    print("wrote", name)


def load_metrics(dataset: str) -> pd.DataFrame:
    df = pickle.load(open(DATA / f"{dataset}-sample-metrics.pkl", "rb"))
    df = df.copy()
    df["dataset"] = dataset
    return df


def dataset_recall(dataset: str) -> pd.Series:
    csv = NB / dataset / "cnn-eval-results" / "per_class_metrics.csv"
    return pd.read_csv(csv, index_col=0)["recall"]


def load_cm(dataset: str) -> pd.DataFrame:
    return pd.read_csv(NB / dataset / "cnn-eval-results" / "confusion_matrix.csv", index_col=0)


# ---------------------------------------------------------------------------
# Dataset characteristics: duration, RMS, centroid
# ---------------------------------------------------------------------------
frames = [load_metrics(p) for p in DATASETS]
all_m = pd.concat(frames, ignore_index=True)

fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.35), layout="constrained")
specs = [
    ("duration_s", "Duration (s)", 1.9, 3.3),
    ("wave_rms_db", "RMS (dB)", -35, -10),
    ("cqt_centroid_hz", "CQT centroid (Hz)", 150, 650),
]
for ax, (col, ylab, ymin, ymax) in zip(axes, specs):
    data = [all_m.loc[all_m.dataset == p, col].to_numpy() for p in DATASETS]
    # clip training -240 dB silence so the other boxes stay readable
    if col == "wave_rms_db":
        data = [np.clip(d, -40, 0) for d in data]
    bp = ax.boxplot(
        data, tick_labels=DATASETS, patch_artist=True, widths=0.62,
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.6},
        capprops={"color": "black", "linewidth": 0.6},
        flierprops={"marker": ".", "markersize": 2, "markerfacecolor": "0.4",
                    "markeredgecolor": "none"},
    )
    for box in bp["boxes"]:
        box.set(facecolor="#d6e3f0", edgecolor="#1f4e79", linewidth=0.7)
    ax.set_ylabel(ylab)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis="x", rotation=35)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
for ax, letter in zip(axes, "ABC"):
    ax.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=8)
save(fig, "esm-dataset-chars")

# ---------------------------------------------------------------------------
# CNN training history (from recorded run in graphs.ipynb)
# ---------------------------------------------------------------------------
epochs = np.arange(0, 10)
train_acc = np.array([0.4765, 0.9841, 0.9904, 0.9943, 0.9947, 0.9931, 0.9952, 0.9991, 0.9976, 0.9976])
val_acc = np.array([0.9290, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985])
train_loss = np.array([1.8565, 0.0511, 0.0297, 0.0177, 0.0157, 0.0220, 0.0137, 0.0040, 0.0064, 0.0057])
val_loss = np.array([0.2188, 0.0030, 0.0056, 0.0014, 0.0020, 0.0095, 0.0071, 0.0065, 0.0066, 0.0066])

fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.15), layout="constrained")
axes[0].plot(epochs, train_acc, color="#1f4e79", linewidth=1.4, label="Train")
axes[0].plot(epochs, val_acc, color="#9a4a12", linewidth=1.4, linestyle="--", label="Validation")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0.4, 1.03)
axes[1].plot(epochs, train_loss, color="#1f4e79", linewidth=1.4, label="Train")
axes[1].plot(epochs, val_loss, color="#9a4a12", linewidth=1.4, linestyle="--", label="Validation")
axes[1].set_ylabel("Loss")
for ax, letter in zip(axes, "AB"):
    ax.set_xlabel("Epoch")
    ax.set_title(f"({letter})", loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
fig.legend(loc="outside lower center", ncol=2)
save(fig, "esm-cnn-history")

# ---------------------------------------------------------------------------
# 5-fold curves
# ---------------------------------------------------------------------------
f1_acc = [0.3623, 0.9790, 0.9919, 0.9938, 0.9969, 0.9967, 0.9986, 0.9985, 0.9992, 0.9988, 0.9979]
f1_val = [0.8663, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
f1_loss = [2.2914, 0.0658, 0.0227, 0.0239, 0.0104, 0.0082, 0.0036, 0.0036, 0.0032, 0.0032, 0.0051]
f2_acc = [0.4460, 0.9855, 0.9909, 0.9950, 0.9944, 0.9975, 0.9985, 0.9979, 0.9975, 0.9986, 0.9979,
          0.9979, 0.9983, 0.9973, 0.9973, 0.9990, 0.9979, 0.9981, 0.9985, 0.9975, 0.9979]
f2_val = [0.9566, 1.0] + [1.0] * 19
f2_loss = [1.9897, 0.0453, 0.0281, 0.0157, 0.0144, 0.0060, 0.0039, 0.0056, 0.0083, 0.0051, 0.0046,
           0.0064, 0.0088, 0.0079, 0.0095, 0.0049, 0.0050, 0.0063, 0.0050, 0.0049, 0.0074]
f3_acc = [0.3372, 0.9705, 0.9869, 0.9911, 0.9959, 0.9977, 0.9965, 0.9977, 0.9986, 0.9975, 0.9967,
          0.9983, 0.9983, 0.9977, 0.9965, 0.9977, 0.9961, 0.9988, 0.9969]
f3_val = [0.6302, 1.0] + [1.0] * 17
f3_loss = [2.3846, 0.0903, 0.0421, 0.0266, 0.0138, 0.0070, 0.0099, 0.0061, 0.0045, 0.0080, 0.0088,
           0.0068, 0.0061, 0.0079, 0.0095, 0.0065, 0.0108, 0.0053, 0.0091]
f4_acc = [0.2733, 0.9464, 0.9780, 0.9855, 0.9917, 0.9923, 0.9958, 0.9952, 0.9959, 0.9969, 0.9963,
          0.9944, 0.9965, 0.9942, 0.9967]
f4_val = [0.7934, 1.0] + [1.0] * 13
f4_loss = [2.5646, 0.1518, 0.0639, 0.0399, 0.0277, 0.0220, 0.0144, 0.0109, 0.0113, 0.0093, 0.0121,
           0.0141, 0.0089, 0.0157, 0.0104]
f5_acc = [0.3476, 0.9799, 0.9890, 0.9954, 0.9967, 0.9983, 0.9988, 0.9988, 0.9986, 0.9985, 0.9992,
          0.9994, 0.9996, 0.9985]
f5_val = [0.9045, 0.9983] + [1.0] * 12
f5_loss = [2.3115, 0.0658, 0.0341, 0.0159, 0.0106, 0.0058, 0.0040, 0.0034, 0.0036, 0.0041, 0.0036,
           0.0019, 0.0024, 0.0058]
folds_acc = [f1_acc, f2_acc, f3_acc, f4_acc, f5_acc]
folds_val = [f1_val, f2_val, f3_val, f4_val, f5_val]
folds_loss = [f1_loss, f2_loss, f3_loss, f4_loss, f5_loss]
cmap = plt.get_cmap("tab10")

fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.45), layout="constrained")
for i, (acc, val) in enumerate(zip(folds_acc, folds_val)):
    ep = np.arange(len(acc))
    axes[0].plot(ep, acc, color=cmap(i), linewidth=1.0, label=f"Fold {i+1}")
    axes[0].plot(ep, val, color=cmap(i), linewidth=1.0, linestyle="--")
for i, loss in enumerate(folds_loss):
    axes[1].plot(np.arange(len(loss)), loss, color=cmap(i), linewidth=1.0, label=f"Fold {i+1}")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0.25, 1.04)
axes[1].set_ylabel("Train loss")
for ax, letter in zip(axes, "AB"):
    ax.set_xlabel("Epoch")
    ax.set_title(f"({letter})", loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
handles = [plt.Line2D([0], [0], color=cmap(i), linewidth=1.2, label=f"Fold {i+1}") for i in range(5)]
handles += [
    plt.Line2D([0], [0], color="0.3", linewidth=1.0, label="Train"),
    plt.Line2D([0], [0], color="0.3", linewidth=1.0, linestyle="--", label="Val"),
]
fig.legend(handles=handles, loc="outside lower center", ncol=7)
save(fig, "esm-cnn-kfold")


def plot_cm(dataset: str, name: str, title: str) -> None:
    cm = load_cm(dataset)
    labels = [pretty_class(c) for c in cm.index]
    mat = cm.to_numpy(dtype=float)
    n = len(labels)
    vmax = max(float(mat.max()), 1.0)
    fig, ax = plt.subplots(figsize=(5.55, 4.85), layout="constrained")
    im = ax.imshow(mat, cmap="Blues", interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(n), labels, rotation=90, fontsize=5.5)
    ax.set_yticks(np.arange(n), labels, fontsize=5.5)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color="0.82", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=8)
    # Annotate nonzero cells only — a wall of zeros is unreadable at 36×36.
    thresh = 0.55 * vmax
    for i, j in zip(*np.nonzero(mat)):
        val = int(mat[i, j])
        ax.text(
            j, i, str(val),
            ha="center", va="center",
            fontsize=3.6,
            color="white" if mat[i, j] >= thresh else "black",
        )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Count")
    save(fig, name)


plot_cm("training", "esm-cm-training", "Training resubstitution (7200 takes)")
plot_cm("vivo", "esm-cm-vivo", "vivo (device)")
plot_cm("flow", "esm-cm-flow", "flow (device)")
plot_cm("thinkpad-2", "esm-cm-thinkpad-2", "thinkpad-2 (keyboard)")
plot_cm("flow-2", "esm-cm-flow-2", "flow-2 (keyboard)")

# ---------------------------------------------------------------------------
# Per-class recall heatmap
# ---------------------------------------------------------------------------
rec = pd.DataFrame({p: dataset_recall(p) for p in OOD})
rec.index = [pretty_class(i) for i in rec.index]
order = rec.min(axis=1).sort_values().index
rec = rec.loc[order]
fig, ax = plt.subplots(figsize=(4.4, 6.4), layout="constrained")
im = ax.imshow(rec.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(np.arange(len(OOD)), OOD, rotation=30, ha="right")
ax.set_yticks(np.arange(len(rec.index)), rec.index, fontsize=6)
ax.set_title("Per-class recall (held-out)")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Recall")
save(fig, "esm-recall-grid")

# ---------------------------------------------------------------------------
# Overlay F1
# ---------------------------------------------------------------------------
noise = pd.read_csv(NB / "aug-noise" / "cnn-compare-results" / "summary_wide.csv").set_index("domain")
rir = pd.read_csv(NB / "aug-rir" / "cnn-compare-results" / "summary_wide.csv").set_index("domain")
dire = pd.read_csv(NB / "aug-dir" / "cnn-compare-results" / "summary_wide.csv").set_index("domain")
series = [
    ("Clean", noise["macro_f1_clean"], dict(facecolor="#d6e3f0", edgecolor="#1f4e79", hatch="")),
    ("Noise", noise["macro_f1_noise"], dict(facecolor="#f6d6d6", edgecolor="#8b1e1e", hatch="///")),
    ("Room IR", rir["macro_f1_rir"], dict(facecolor="#f6e0cc", edgecolor="#9a4a12", hatch="xxx")),
    ("Mic IR", dire["macro_f1_dir"], dict(facecolor="#d9efe4", edgecolor="#1b6b45", hatch="...")),
]
fig, ax = plt.subplots(figsize=(6.85, 2.35), layout="constrained")
x = np.arange(len(OOD))
width = 0.18
off = (np.arange(4) - 1.5) * width
for k, (lab, ser, st) in enumerate(series):
    vals = [float(ser.loc[p]) for p in OOD]
    ax.bar(x + off[k], vals, width=width * 0.95,
           facecolor=st["facecolor"], edgecolor=st["edgecolor"],
           linewidth=0.7, hatch=st["hatch"], label=lab)
ax.set_xticks(x, OOD)
ax.set_ylabel("Macro F1")
ax.set_ylim(0.78, 1.02)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
fig.legend(loc="outside lower center", ncol=4)
save(fig, "esm-overlay-f1")

# ---------------------------------------------------------------------------
# Overlay CQT: same clip, clean / noise / RIR / DIR
# ---------------------------------------------------------------------------
OVERLAY_NPZ = {
    "Clean": "{p}.npz",
    "Noise": "{p}-noise-interior_domestic.npz",
    "Room IR": "{p}-rir-indoor_no_bathroom.npz",
    "Mic IR": "{p}-dir-mic6.npz",
}
demo_pack = "vivo"
demo_classes = ["C_major_4", "E_diminished_4"]
banks = {lab: np.load(FEAT / spec.format(p=demo_pack)) for lab, spec in OVERLAY_NPZ.items()}
labs = np.asarray(banks["Clean"]["labels"])
fig, axes = plt.subplots(2, 4, figsize=(6.85, 3.55), layout="constrained", sharex=True, sharey=True)
for r, cls in enumerate(demo_classes):
    idx = int(np.flatnonzero(labs == cls)[0])
    for c, lab in enumerate(OVERLAY_NPZ):
        ax = axes[r, c]
        ax.imshow(banks[lab]["features"][idx], origin="lower", aspect="auto", cmap="magma")
        if r == 0:
            ax.set_title(lab, fontsize=8)
        if c == 0:
            ax.set_ylabel(f"{pretty_class(cls)}\nCQT bin")
        if r == 1:
            ax.set_xlabel("Frame")
save(fig, "esm-overlay-cqt")

# ---------------------------------------------------------------------------
# Overlay recall deltas (classes that move)
# ---------------------------------------------------------------------------
delta_frames = []
for tag, col, path in [
    ("Noise", "delta_recall", NB / "aug-noise" / "cnn-compare-results" / "perclass_comparison.csv"),
    ("Room IR", "delta_recall", NB / "aug-rir" / "cnn-compare-results" / "perclass_comparison.csv"),
    ("Mic IR", "delta_recall", NB / "aug-dir" / "cnn-compare-results" / "perclass_comparison.csv"),
]:
    d = pd.read_csv(path)
    d["overlay"] = tag
    delta_frames.append(d)
deltas = pd.concat(delta_frames, ignore_index=True)
# keep classes with a real move on any overlay/dataset
move = deltas.groupby("class")["delta_recall"].apply(lambda s: s.abs().max() >= 0.10)
keep_cls = move[move].index
fig, axes = plt.subplots(1, 3, figsize=(6.85, 4.4), layout="constrained", sharey=True)
vmax = 0.5
im = None
for ax, tag in zip(axes, ["Noise", "Room IR", "Mic IR"]):
    sub = deltas[deltas.overlay == tag]
    mat = sub.pivot(index="class", columns="domain", values="delta_recall")
    mat = mat.reindex(index=keep_cls, columns=OOD)
    mat.index = [pretty_class(i) for i in mat.index]
    # worst drop at top
    mat = mat.loc[mat.min(axis=1).sort_values().index]
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(OOD)), OOD, rotation=35, ha="right", fontsize=6)
    ax.set_yticks(np.arange(len(mat.index)), mat.index, fontsize=6)
    ax.set_title(tag, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.to_numpy()[i, j]
            if abs(v) >= 0.05:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=4.2,
                        color="white" if abs(v) > 0.28 else "black")
fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="Recall delta (degraded − clean)")
save(fig, "esm-overlay-delta")

# ---------------------------------------------------------------------------
# Yamaha clean vs noise recall (classes that move)
# ---------------------------------------------------------------------------
pc = pd.read_csv(NB / "aug-noise" / "cnn-compare-results" / "perclass_comparison.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.8), layout="constrained")
for ax, dataset, letter in zip(axes, ["thinkpad-2", "flow-2"], "AB"):
    sub = pc[pc.domain == dataset].copy()
    sub["pretty"] = sub["class"].map(pretty_class)
    keep = sub[(sub.clean < 0.95) | (sub.noise < 0.85)].sort_values("noise")
    y = np.arange(len(keep))
    h = 0.36
    ax.barh(y - h / 2, keep.clean, height=h,
            facecolor="#d6e3f0", edgecolor="#1f4e79", linewidth=0.6)
    ax.barh(y + h / 2, keep.noise, height=h,
            facecolor="#f6d6d6", edgecolor="#8b1e1e", linewidth=0.6, hatch="///")
    ax.set_yticks(y, keep.pretty, fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_title(f"({letter})  {dataset}", loc="left", fontweight="bold")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")
fig.legend(
    handles=[
        Patch(facecolor="#d6e3f0", edgecolor="#1f4e79", label="Clean"),
        Patch(facecolor="#f6d6d6", edgecolor="#8b1e1e", hatch="///", label="Additive noise"),
    ],
    loc="outside lower center", ncol=2,
)
save(fig, "esm-yamaha-noise")

# ---------------------------------------------------------------------------
# Envelope shift: per-clip dB level distribution, and what per-clip centering
# explains of each class centroid's drift from training (offset / scale / residual)
# ---------------------------------------------------------------------------
BASELINE_RESULTS = NB / "baseline-centroid" / "baseline-results"
envelope = pd.read_csv(BASELINE_RESULTS / "envelope_stats.csv")
decomp = pd.read_csv(BASELINE_RESULTS / "envelope_drift_decomposition.csv")

ENV_DATASETS = ["training"] + OOD
fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.6), layout="constrained")

ax = axes[0]
data = [envelope.loc[envelope.dataset == p, "clip_mean_db"].to_numpy() for p in ENV_DATASETS]
bp = ax.boxplot(
    data, tick_labels=ENV_DATASETS, patch_artist=True, widths=0.62, showfliers=False,
    medianprops={"color": "black", "linewidth": 0.8},
    whiskerprops={"color": "black", "linewidth": 0.6},
    capprops={"color": "black", "linewidth": 0.6},
)
for box in bp["boxes"]:
    box.set(facecolor="#d6e3f0", edgecolor="#1f4e79", linewidth=0.7)
ax.set_ylabel("Per-clip mean CQT level (dB)")
ax.tick_params(axis="x", rotation=30)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75")

ax = axes[1]
agg = decomp.groupby("dataset")[["frac_offset", "frac_scale", "frac_residual"]].mean().loc[OOD]
x = np.arange(len(OOD))
parts = [
    ("frac_offset", "Offset (mean-centering)", "", "#1f4e79"),
    ("frac_scale", "Scale (unit-normalizing)", "..", "#1b6b45"),
    ("frac_residual", "Residual (not removed)", None, "0.85"),
]
bottom = np.zeros(len(OOD))
for col, lab, hatch, color in parts:
    face = color if hatch is None else "white"
    edge = "0.4" if hatch is None else color
    ax.bar(x, agg[col], bottom=bottom, label=lab, facecolor=face, edgecolor=edge,
           linewidth=0.7, hatch=hatch)
    bottom += agg[col].to_numpy()
ax.set_xticks(x, OOD, rotation=30, ha="right")
ax.set_ylabel("Share of class-centroid drift")
ax.set_ylim(0, 1.02)
for ax_, letter in zip(axes, "AB"):
    ax_.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=8)
fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=3)
save(fig, "esm-envelope-shift")

# ---------------------------------------------------------------------------
# Mean CQT difference vs training
# ---------------------------------------------------------------------------
train = np.load(FEAT / "training.npz")
train_mean = train["features"].mean(axis=0)
fig, axes = plt.subplots(2, 3, figsize=(6.85, 3.6), layout="constrained")
axes = axes.ravel()
vmax = 0
diffs = {}
for p in OOD:
    z = np.load(FEAT / f"{p}.npz")
    d = z["features"].mean(axis=0) - train_mean
    diffs[p] = d
    vmax = max(vmax, float(np.percentile(np.abs(d), 99)))
vmax = max(vmax, 1.0)
for ax, p in zip(axes, OOD):
    im = ax.imshow(diffs[p], origin="lower", aspect="auto", cmap="coolwarm",
                   vmin=-vmax, vmax=vmax)
    ax.set_title(p, fontsize=8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("CQT bin")
axes[-1].axis("off")
fig.colorbar(im, ax=axes[:5], fraction=0.03, pad=0.02, label="OOD mean − train mean (dB)")
save(fig, "esm-cqt-shift")
print("done")
