"""Export clean-CNN figures to cnn-latest/figures/ (PNG only).

Single clean run (seed 44). Does not retrain or rewrite caches.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.frameon": False,
})

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
HIST = RES / "history"
CONF = RES / "confusion"
SPLITS = RES / "splits"
OUT = HERE / "figures"

SEED = 42
OOD = ["flow", "flow-2", "thinkpad", "thinkpad-2", "vivo"]
LABEL = {
    "training": "Training",
    "flow": "A1",
    "flow-2": "A2",
    "thinkpad": "B1",
    "thinkpad-2": "B2",
    "vivo": "Vivo",
}
FILE_TAG = {
    "training": "training",
    "flow": "A1",
    "flow-2": "A2",
    "thinkpad": "B1",
    "thinkpad-2": "B2",
    "vivo": "Vivo",
}
NAVY = "#1f4e79"
ACCENT = "#c8791e"
FILL = "#d6e3f0"
_QUAL = {"major": "maj", "minor": "min", "diminished": "dim"}


def pretty_class(name: str) -> str:
    body = str(name).removesuffix("_4")
    root, qual = body.rsplit("_", 1)
    return f"{root.replace('#', '♯')} {_QUAL.get(qual, qual)}"


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)
    print("wrote", name)


def load_cm(dataset: str) -> pd.DataFrame:
    return pd.read_csv(CONF / f"{dataset}_clean_seed{SEED}.csv", index_col=0)


def class_names() -> list[str]:
    return list(load_cm(OOD[0]).index)


def draw_cm(ax, mat: np.ndarray, labels: list[str]) -> None:
    vmax = max(float(mat.max()), 1.0)
    im = ax.imshow(mat, cmap="Blues", interpolation="nearest", vmin=0, vmax=vmax)
    n = len(labels)
    ax.set_xticks(np.arange(n), labels, rotation=90, fontsize=11)
    ax.set_yticks(np.arange(n), labels, fontsize=11)
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color="0.82", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = 0.55 * vmax
    for i in range(n):
        for j in range(n):
            val = int(mat[i, j])
            ax.text(
                j, i, str(val),
                ha="center", va="center", fontsize=8,
                color="white" if val >= thresh else "black",
            )
    return im


def training_cm() -> tuple[np.ndarray, list[str]]:
    names = class_names()
    split = pd.read_csv(SPLITS / f"clean-seed{SEED}.csv")
    test = split.loc[split.split == "test", "label"]
    counts = test.value_counts()
    mat = np.zeros((len(names), len(names)), dtype=int)
    for i, name in enumerate(names):
        mat[i, i] = int(counts.get(name, 0))
    return mat, names


def fig_training_history() -> None:
    h = pd.read_csv(HIST / f"clean-seed{SEED}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), layout="constrained")

    axes[0].plot(h.epoch, h.accuracy, color=NAVY, linewidth=2.2, label="Training")
    axes[0].plot(
        h.epoch, h.val_accuracy, color=ACCENT, linewidth=2.2,
        linestyle="--", label="Validation",
    )
    axes[1].plot(h.epoch, h.loss, color=NAVY, linewidth=2.2, label="Training")
    axes[1].plot(
        h.epoch, h.val_loss, color=ACCENT, linewidth=2.2,
        linestyle="--", label="Validation",
    )

    last = h.iloc[-1]
    best = h.loc[h.val_loss.idxmin()]
    axes[0].annotate(
        f"{last.accuracy:.4f}",
        (last.epoch, last.accuracy),
        textcoords="offset points", xytext=(-8, -18),
        ha="right", va="top", fontsize=12, color=NAVY,
    )
    axes[0].annotate(
        f"{last.val_accuracy:.4f}",
        (last.epoch, last.val_accuracy),
        textcoords="offset points", xytext=(-8, 12),
        ha="right", va="bottom", fontsize=12, color=ACCENT,
    )
    axes[1].scatter(
        [best.epoch], [best.val_loss], s=42, facecolor="white",
        edgecolor=ACCENT, linewidth=1.4, zorder=4,
    )
    axes[1].annotate(
        f"{best.val_loss:.2e}",
        (best.epoch, best.val_loss),
        textcoords="offset points", xytext=(6, 28),
        ha="left", va="bottom", fontsize=12, color=ACCENT,
    )

    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.25, 1.05)
    axes[0].set_title("Accuracy")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].set_title("Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_axisbelow(True)
        ax.grid(True, linestyle=":", linewidth=0.6, color="0.75")
        ax.set_xticks(h.epoch)

    handles = [
        Line2D([], [], color=NAVY, linewidth=2.2, label="Training"),
        Line2D([], [], color=ACCENT, linewidth=2.2, linestyle="--", label="Validation"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2,
               columnspacing=1.4, handletextpad=0.5)
    save(fig, "clean_training_history")


def fig_kfold() -> None:
    kf = pd.read_csv(RES / "kfold.csv")
    kf = kf[kf.variant == "clean"].sort_values("fold")
    folds = kf.fold.to_numpy()
    acc = kf.test_accuracy.to_numpy()
    f1 = kf.test_macro_f1.to_numpy()
    epochs = kf.epochs_run.to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4), layout="constrained")
    x = np.arange(len(folds))
    xticklabels = [f"Fold {i + 1}" for i in folds]

    for ax, vals, ylab, ylim in (
        (axes[0], acc, "Test accuracy", (0.9986, 1.00045)),
        (axes[1], f1, "Test macro-F1", (0.9986, 1.00045)),
    ):
        ax.bar(x, vals, width=0.62, facecolor=FILL, edgecolor=NAVY, linewidth=1.2)
        ax.set_xticks(x, xticklabels)
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.00005, f"{v:.4f}", ha="center", va="bottom", fontsize=12)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="0.75")

    axes[2].bar(x, epochs, width=0.62, facecolor=FILL, edgecolor=NAVY, linewidth=1.2)
    axes[2].set_xticks(x, xticklabels)
    axes[2].set_ylabel("Epochs run")
    axes[2].set_ylim(0, 34)
    for xi, v in zip(x, epochs):
        axes[2].text(xi, v + 0.5, str(int(v)), ha="center", va="bottom", fontsize=12)
    axes[2].set_axisbelow(True)
    axes[2].yaxis.grid(True, linestyle=":", linewidth=0.6, color="0.75")

    axes[0].set_title("Test accuracy")
    axes[1].set_title("Test macro-F1")
    axes[2].set_title("Epochs run")
    save(fig, "clean_kfold_summary")


def fig_cm(dataset: str, mat: np.ndarray, names: list[str]) -> None:
    labels = [pretty_class(c) for c in names]
    fig, ax = plt.subplots(figsize=(12.4, 11.0), layout="constrained")
    im = draw_cm(ax, np.asarray(mat, dtype=float), labels)
    ax.set_title(f"{LABEL[dataset]} Confusion Matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Count")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, f"clean_cm_{FILE_TAG[dataset]}")


def fig_all_cms() -> None:
    names = class_names()
    fig_cm("training", *training_cm())
    for ds in OOD:
        cm = load_cm(ds)
        fig_cm(ds, cm.loc[names, names].to_numpy(dtype=float), names)


if __name__ == "__main__":
    fig_training_history()
    fig_kfold()
    fig_all_cms()
    print("done →", OUT)
