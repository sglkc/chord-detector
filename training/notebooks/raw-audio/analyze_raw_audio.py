#!/usr/bin/env python3
"""Raw-audio characteristics across training / flow / flow-2 / thinkpad / thinkpad-2.

No CQT. Waveform + STFT scalars only. Writes a parquet cache and a small
figure/table pack under results/.
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]  # training/
DATASETS_DIR = ROOT / "datasets"
OUT_DIR = Path(__file__).resolve().parent / "results"

DATASETS = [
    {
        "id": "training",
        "label": "training",
        "short": "Train",
        "keyboard": "Techno T-9890i",
        "mic": "Redmi Pad 2",
        "ext": ".ogg",
    },
    {
        "id": "flow",
        "label": "A1",
        "short": "A1",
        "keyboard": "Techno T-9890i",
        "mic": "ROG Flow X13",
        "ext": ".wav",
    },
    {
        "id": "flow-2",
        "label": "A2",
        "short": "A2",
        "keyboard": "Yamaha PSR-S910",
        "mic": "ROG Flow X13",
        "ext": ".wav",
    },
    {
        "id": "thinkpad",
        "label": "B1",
        "short": "B1",
        "keyboard": "Techno T-9890i",
        "mic": "ThinkPad P14s",
        "ext": ".wav",
    },
    {
        "id": "thinkpad-2",
        "label": "B2",
        "short": "B2",
        "keyboard": "Yamaha PSR-S910",
        "mic": "ThinkPad P14s",
        "ext": ".wav",
    },
]
DS_BY_ID = {d["id"]: d for d in DATASETS}
ORDER = [d["id"] for d in DATASETS]
LABELS = {d["id"]: d["label"] for d in DATASETS}
COLORS = {
    "training": "#4d4d4d",
    "flow": "#1f77b4",
    "flow-2": "#9ecae1",
    "thinkpad": "#d95f02",
    "thinkpad-2": "#fdcdac",
}

SR_TARGET = 48000
N_FFT = 2048
HOP = 512
N_MFCC = 13
SILENCE_DB = -40.0
BANDS = {
    "band_rumble": (20.0, 80.0),
    "band_low": (80.0, 250.0),
    "band_fund": (250.0, 800.0),
    "band_mid": (800.0, 2500.0),
    "band_presence": (2500.0, 6000.0),
    "band_air": (6000.0, 16000.0),
}
CONTRASTS = [
    ("flow", "flow-2", "keyboard @ Flow"),
    ("thinkpad", "thinkpad-2", "keyboard @ ThinkPad"),
    ("flow", "thinkpad", "mic @ Techno"),
    ("flow-2", "thinkpad-2", "mic @ Yamaha"),
    ("training", "flow", "tablet vs Flow @ Techno"),
    ("training", "thinkpad", "tablet vs ThinkPad @ Techno"),
]
TIMBRE_COLS = [
    "centroid_hz",
    "bandwidth_hz",
    "rolloff_hz",
    "log_flatness",
    "slope",
    "hf_db",
    "zcr",
    "crest",
    "attack_s",
    "decay_s",
    "onset_s",
    "temporal_centroid",
    "band_rumble",
    "band_low",
    "band_fund",
    "band_mid",
    "band_presence",
    "band_air",
    "band_mid_log",
    "band_presence_log",
    "band_air_log",
] + [f"mfcc_{i}" for i in range(1, N_MFCC)]


def parse_class(name: str) -> tuple[str, str]:
    # e.g. A#_diminished_4
    parts = name.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0], parts[1]
    return name, ""


def list_files(ds_id: str, ext: str, per_class: int | None, seed: int) -> list[Path]:
    root = DATASETS_DIR / ds_id
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    out: list[Path] = []
    rng = np.random.default_rng(seed)
    for cls in classes:
        files = sorted(cls.glob(f"*{ext}"))
        if per_class is not None and len(files) > per_class:
            idx = rng.choice(len(files), size=per_class, replace=False)
            files = [files[i] for i in sorted(idx)]
        out.extend(files)
    return out


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return float32 array shaped (n_ch, n) and native sample rate."""
    suffix = path.suffix.lower()
    if suffix == ".wav":
        y, sr = sf.read(str(path), always_2d=True, dtype="float32")
        return y.T, int(sr)
    import librosa

    y, sr = librosa.load(str(path), sr=None, mono=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    return y, int(sr)


def _stft_mag(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    n_frames = 1 + (y.size - n_fft) // hop
    if n_frames < 1:
        frame = np.pad(y, (0, n_fft - y.size)) * window
        return np.abs(np.fft.rfft(frame)).astype(np.float32)[:, None]
    frames = sliding_window_view(y, n_fft)[::hop]
    frames = frames * window
    return np.abs(np.fft.rfft(frames, axis=1)).T.astype(np.float32)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    mels = np.linspace(_hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([sr / 2.0]))[0], n_mels + 2)
    hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        right = min(right, n_bins - 1)
        center = min(center, n_bins - 2)
        if center > left:
            fb[m, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[m, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def _dct_ii(n_mfcc: int, n_mels: int) -> np.ndarray:
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    basis = np.cos(np.pi * k * (2 * n + 1) / (2.0 * n_mels))
    basis *= np.sqrt(2.0 / n_mels)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis.astype(np.float32)


_MEL_FB = None
_DCT = None


def _mfcc_from_power(power: np.ndarray, sr: int) -> np.ndarray:
    global _MEL_FB, _DCT
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(sr, N_FFT, n_mels=40)
        _DCT = _dct_ii(N_MFCC, 40)
    mel = _MEL_FB @ power
    log_mel = np.log(mel + 1e-12)
    return _DCT @ log_mel


def extract_one(path_str: str) -> dict:
    path = Path(path_str)
    class_name = path.parent.name
    root, quality = parse_class(class_name)
    rec: dict = {
        "path": str(path),
        "filename": path.name,
        "class_name": class_name,
        "root": root,
        "quality": quality,
        "ok": False,
    }
    try:
        y_st, sr = load_audio(path)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = str(exc)
        return rec

    n_ch, n = y_st.shape
    if sr != SR_TARGET:
        rec["error"] = f"sr={sr}"
        rec["sr"] = sr
        rec["n_channels"] = n_ch
        return rec

    y = y_st.mean(axis=0) if n_ch > 1 else y_st[0]
    peak = float(np.max(np.abs(y)))
    rec.update(
        {
            "sr": sr,
            "n_channels": n_ch,
            "n_samples": int(n),
            "duration_s": n / sr,
            "peak": peak,
            "peak_db": float(20.0 * np.log10(peak + 1e-12)),
        }
    )
    if peak < 1e-8:
        rec["ok"] = True
        rec["silent"] = True
        rec["rms_db"] = -240.0
        return rec

    rec["silent"] = False
    rms = float(np.sqrt(np.mean(y * y)))
    rec["rms"] = rms
    rec["rms_db"] = float(20.0 * np.log10(rms + 1e-12))
    rec["crest"] = peak / max(rms, 1e-12)
    rec["dc"] = float(y.mean())
    rec["zcr"] = float(np.mean(np.abs(np.diff(np.signbit(y)))))
    rec["clip_frac"] = float(np.mean(np.abs(y) >= 0.99))

    if n_ch == 2:
        left, right = y_st[0], y_st[1]
        rec["ch_corr"] = float(np.corrcoef(left, right)[0, 1])
        rms_l = float(np.sqrt(np.mean(left * left)) + 1e-12)
        rms_r = float(np.sqrt(np.mean(right * right)) + 1e-12)
        rec["lr_rms_db"] = float(20.0 * np.log10(rms_l / rms_r))
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        rec["mid_side_db"] = float(
            20.0 * np.log10((np.sqrt(np.mean(mid * mid)) + 1e-12) / (np.sqrt(np.mean(side * side)) + 1e-12))
        )
    else:
        rec["ch_corr"] = np.nan
        rec["lr_rms_db"] = np.nan
        rec["mid_side_db"] = np.nan

    # Frame RMS envelope (same hop as STFT).
    if n < N_FFT:
        y_pad = np.pad(y, (0, N_FFT - n))
    else:
        y_pad = y
    env = np.sqrt(np.mean(sliding_window_view(y_pad, N_FFT)[::HOP] ** 2, axis=1)).astype(np.float32)
    n_frames = int(env.size)
    env_db = 20.0 * np.log10(env + 1e-12)
    env_peak = float(env.max())
    env_peak_db = 20.0 * np.log10(env_peak + 1e-12)
    active = env_db > (env_peak_db + SILENCE_DB)
    rec["silence_ratio"] = float(1.0 - active.mean())
    if active.any():
        first, last = int(np.argmax(active)), int(len(active) - 1 - np.argmax(active[::-1]))
        rec["lead_silence_s"] = first * HOP / sr
        rec["trail_silence_s"] = (n_frames - 1 - last) * HOP / sr
        rec["active_s"] = (last - first + 1) * HOP / sr
        noise = env[: max(first, 1)]
    else:
        rec["lead_silence_s"] = rec["duration_s"]
        rec["trail_silence_s"] = 0.0
        rec["active_s"] = 0.0
        noise = env[:1]
    rec["noise_floor_db"] = float(20.0 * np.log10(float(np.median(noise)) + 1e-12))
    rec["snr_db"] = rec["rms_db"] - rec["noise_floor_db"]
    n20 = max(int(0.020 * sr), 1)
    rec["first_20ms_db"] = float(20.0 * np.log10(float(np.sqrt(np.mean(y[:n20] ** 2))) + 1e-12))
    # First-frame "noise" is only a floor when the note has not started yet.
    if rec["lead_silence_s"] < (HOP / sr) or rec["snr_db"] < 12.0:
        rec["noise_floor_db"] = np.nan
        rec["snr_db"] = np.nan

    peak_i = int(np.argmax(env))
    rec["peak_pos"] = peak_i / max(n_frames - 1, 1)
    thr10, thr90 = 0.10 * env_peak, 0.90 * env_peak
    pre = env[: peak_i + 1]
    i10 = int(np.argmax(pre >= thr10)) if (pre >= thr10).any() else 0
    i90 = int(np.argmax(pre >= thr90)) if (pre >= thr90).any() else peak_i
    rec["attack_s"] = max(i90 - i10, 0) * HOP / sr
    rec["onset_s"] = i10 * HOP / sr
    post = env[peak_i:]
    target = env_peak / np.e
    decay_i = int(np.argmax(post <= target)) if (post <= target).any() else (len(post) - 1)
    rec["decay_s"] = decay_i * HOP / sr
    times = (np.arange(n_frames) * HOP + N_FFT / 2.0) / sr
    rec["temporal_centroid"] = float(np.sum(times * env) / (np.sum(env) + 1e-12) / rec["duration_s"])

    mag = _stft_mag(y, N_FFT, HOP)
    power = mag * mag
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr).astype(np.float32)
    p_frame = power + 1e-20
    p_sum = p_frame.sum(axis=0, keepdims=True)
    centroid = (freqs[:, None] * p_frame).sum(axis=0) / p_sum[0]
    rec["centroid_hz"] = float(centroid.mean())
    rec["bandwidth_hz"] = float(
        np.sqrt((((freqs[:, None] - centroid) ** 2) * p_frame).sum(axis=0) / p_sum[0]).mean()
    )
    csum = np.cumsum(p_frame, axis=0)
    roll_idx = np.argmax(csum >= 0.85 * p_sum, axis=0)
    rec["rolloff_hz"] = float(freqs[roll_idx].mean())
    rec["flatness"] = float(
        np.exp(np.mean(np.log(p_frame), axis=0)).mean() / (p_frame.mean(axis=0).mean() + 1e-20)
    )
    rec["log_flatness"] = float(np.log10(rec["flatness"] + 1e-20))
    # Spectral slope on the clip-mean log spectrum vs log-frequency (200 Hz–8 kHz).
    mean_p = p_frame.mean(axis=1)
    mask = (freqs >= 200.0) & (freqs <= 8000.0)
    x = np.log10(freqs[mask])
    yy = np.log10(mean_p[mask] + 1e-20)
    rec["slope"] = float(np.polyfit(x, yy, 1)[0])
    rec["hf_ratio"] = float(mean_p[freqs >= 4000.0].sum() / (mean_p.sum() + 1e-20))
    rec["hf_db"] = float(10.0 * np.log10(rec["hf_ratio"] + 1e-12))
    total = float(mean_p.sum())
    log_mean = np.log10(mean_p + 1e-20)
    for name, (lo, hi) in BANDS.items():
        band = (freqs >= lo) & (freqs < hi)
        frac = float(mean_p[band].sum() / (total + 1e-20))
        rec[name] = frac
        rec[f"{name}_db"] = float(10.0 * np.log10(frac + 1e-12))
        rec[f"{name}_log"] = float(log_mean[band].mean()) if band.any() else np.nan

    mfcc = _mfcc_from_power(p_frame, sr)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    for i in range(N_MFCC):
        rec[f"mfcc_{i}"] = float(mfcc_mean[i])
        rec[f"mfcc_{i}_std"] = float(mfcc_std[i])

    # Energy-normalized mean log spectrum for later averaging.
    spec = mean_p / (mean_p.sum() + 1e-20)
    rec["_spec"] = np.log10(spec + 1e-16).astype(np.float32)
    rec["ok"] = True
    return rec


def _init_worker() -> None:
    global _MEL_FB, _DCT
    _MEL_FB = None
    _DCT = None


def extract_dataset(
    ds: dict,
    per_class: int | None,
    seed: int,
    workers: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    files = list_files(ds["id"], ds["ext"], per_class, seed)
    rows = []
    specs = []
    if workers <= 1:
        for p in files:
            rec = extract_one(str(p))
            spec = rec.pop("_spec", None)
            rec["dataset"] = ds["id"]
            rec["label"] = ds["label"]
            rec["keyboard"] = ds["keyboard"]
            rec["mic"] = ds["mic"]
            rows.append(rec)
            if spec is not None:
                specs.append(spec)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futs = {ex.submit(extract_one, str(p)): p for p in files}
            for fut in as_completed(futs):
                rec = fut.result()
                spec = rec.pop("_spec", None)
                rec["dataset"] = ds["id"]
                rec["label"] = ds["label"]
                rec["keyboard"] = ds["keyboard"]
                rec["mic"] = ds["mic"]
                rows.append(rec)
                if spec is not None:
                    specs.append(spec)
    frame = pd.DataFrame(rows)
    n_bins = N_FFT // 2 + 1
    if specs:
        spec_mean = np.mean(np.stack(specs, axis=0), axis=0)
    else:
        spec_mean = np.full(n_bins, np.nan, dtype=np.float32)
    return frame, spec_mean


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "n",
        "n_silent",
        "duration_s",
        "rms_db",
        "peak_db",
        "crest",
        "zcr",
        "centroid_hz",
        "bandwidth_hz",
        "rolloff_hz",
        "flatness",
        "log_flatness",
        "slope",
        "hf_ratio",
        "hf_db",
        "attack_s",
        "decay_s",
        "onset_s",
        "first_20ms_db",
        "noise_floor_db",
        "snr_db",
        "ch_corr",
        "mid_side_db",
        "band_rumble",
        "band_low",
        "band_fund",
        "band_mid",
        "band_presence",
        "band_air",
        "mfcc_1",
        "mfcc_2",
    ]
    rows = []
    for ds_id in ORDER:
        sub = df[df.dataset == ds_id]
        ok = sub[sub.ok & ~sub.silent.fillna(False)]
        row = {
            "dataset": ds_id,
            "label": LABELS[ds_id],
            "keyboard": DS_BY_ID[ds_id]["keyboard"],
            "mic": DS_BY_ID[ds_id]["mic"],
            "n": int(len(sub)),
            "n_silent": int(sub.silent.fillna(False).sum()),
            "n_channels": ok["n_channels"].mode().iloc[0] if len(ok) else np.nan,
        }
        for c in cols[2:]:
            if c in ok.columns:
                row[c] = float(ok[c].mean())
                row[f"{c}_std"] = float(ok[c].std())
        rows.append(row)
    return pd.DataFrame(rows)


def contrast_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    ok = df[df.ok & ~df.silent.fillna(False)]
    for a, b, name in CONTRASTS:
        aa, bb = ok[ok.dataset == a], ok[ok.dataset == b]
        for col in cols:
            if col not in ok.columns:
                continue
            d = cohens_d(aa[col].to_numpy(), bb[col].to_numpy())
            rows.append(
                {
                    "contrast": name,
                    "a": LABELS[a],
                    "b": LABELS[b],
                    "feature": col,
                    "mean_a": float(aa[col].mean()),
                    "mean_b": float(bb[col].mean()),
                    "delta": float(aa[col].mean() - bb[col].mean()),
                    "cohens_d": d,
                    "abs_d": abs(d),
                }
            )
    return pd.DataFrame(rows)


def classify(df: pd.DataFrame, cols: list[str], y_col: str = "dataset") -> pd.DataFrame:
    ok = df[df.ok & ~df.silent.fillna(False)].copy()
    X = ok[cols].to_numpy()
    y = ok[y_col].to_numpy()
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    chance = 1.0 / len(np.unique(y))
    return pd.DataFrame(
        [
            {
                "target": y_col,
                "n": int(len(y)),
                "n_classes": int(len(np.unique(y))),
                "chance": chance,
                "acc_mean": float(scores.mean()),
                "acc_std": float(scores.std()),
            }
        ]
    )


def pairwise_acc(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    ok = df[df.ok & ~df.silent.fillna(False)]
    for a, b, name in CONTRASTS:
        sub = ok[ok.dataset.isin([a, b])]
        X = sub[cols].to_numpy()
        y = sub["dataset"].to_numpy()
        mask = np.isfinite(X).all(axis=1)
        X, y = X[mask], y[mask]
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        rows.append(
            {
                "contrast": name,
                "a": LABELS[a],
                "b": LABELS[b],
                "n": int(len(y)),
                "acc_mean": float(scores.mean()),
                "acc_std": float(scores.std()),
            }
        )
    return pd.DataFrame(rows)


def style_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="0.75")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def boxplot_panel(df: pd.DataFrame, specs: list[tuple[str, str, float | None, float | None]], path: Path) -> None:
    ok = df[df.ok & ~df.silent.fillna(False)]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.2), layout="constrained")
    for ax, (col, ylab, ymin, ymax) in zip(axes.ravel(), specs):
        data = [ok.loc[ok.dataset == ds, col].dropna().to_numpy() for ds in ORDER]
        bp = ax.boxplot(
            data,
            tick_labels=[LABELS[d] for d in ORDER],
            patch_artist=True,
            widths=0.65,
            medianprops={"color": "black", "linewidth": 0.8},
            flierprops={"marker": ".", "markersize": 2, "markerfacecolor": "0.45", "markeredgecolor": "none"},
        )
        for box, ds in zip(bp["boxes"], ORDER):
            box.set(facecolor=COLORS[ds], edgecolor="0.2", linewidth=0.7)
        ax.set_ylabel(ylab)
        if ymin is not None or ymax is not None:
            ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="x", rotation=30)
        style_axes(ax)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_spectra(spec_map: dict[str, np.ndarray], path: Path) -> None:
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR_TARGET)
    mask = (freqs >= 50.0) & (freqs <= 16000.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), layout="constrained")

    ax = axes[0]
    for ds in ORDER:
        spec = spec_map[ds]
        ax.plot(freqs[mask], spec[mask], color=COLORS[ds], lw=1.6, label=LABELS[ds])
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean log10 energy-normalized STFT")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    ax.set_title("All five")

    ax = axes[1]
    # Keyboard residual: Yamaha minus Techno on the same mic.
    ax.plot(
        freqs[mask],
        spec_map["flow-2"][mask] - spec_map["flow"][mask],
        color="#1f77b4",
        lw=1.6,
        label="A2 − A1  (Yamaha − Techno @ Flow)",
    )
    ax.plot(
        freqs[mask],
        spec_map["thinkpad-2"][mask] - spec_map["thinkpad"][mask],
        color="#d95f02",
        lw=1.6,
        label="B2 − B1  (Yamaha − Techno @ ThinkPad)",
    )
    ax.axhline(0.0, color="0.4", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Δ log10 spectrum")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    ax.set_title("Keyboard residual (same mic)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pca(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    ok = df[df.ok & ~df.silent.fillna(False)].copy()
    X = ok[cols].to_numpy()
    mask = np.isfinite(X).all(axis=1)
    ok = ok.loc[mask]
    X = StandardScaler().fit_transform(X[mask])
    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.4, 5.0), layout="constrained")
    for ds in ORDER:
        m = ok.dataset.to_numpy() == ds
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.45, color=COLORS[ds], label=LABELS[ds], linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, markerscale=2)
    style_axes(ax)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_bands(df: pd.DataFrame, path: Path) -> None:
    ok = df[df.ok & ~df.silent.fillna(False)]
    band_cols = list(BANDS)
    means = ok.groupby("dataset")[band_cols].mean().reindex(ORDER)
    fig, ax = plt.subplots(figsize=(8.6, 4.2), layout="constrained")
    x = np.arange(len(band_cols))
    width = 0.16
    for i, ds in enumerate(ORDER):
        ax.bar(
            x + (i - 2) * width,
            means.loc[ds].to_numpy(),
            width=width,
            color=COLORS[ds],
            edgecolor="0.2",
            linewidth=0.4,
            label=LABELS[ds],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["20–80", "80–250", "250–800", "0.8–2.5k", "2.5–6k", "6–16k"])
    ax.set_ylabel("Mean energy fraction")
    ax.set_xlabel("Hz")
    ax.legend(frameon=False, ncol=5, fontsize=8)
    style_axes(ax)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_effect_heatmap(effects: pd.DataFrame, path: Path) -> None:
    feats = [
        "rms_db",
        "centroid_hz",
        "slope",
        "hf_db",
        "log_flatness",
        "onset_s",
        "decay_s",
        "first_20ms_db",
        "crest",
        "band_low",
        "band_mid",
        "mfcc_1",
    ]
    contrasts = [c for _, _, c in CONTRASTS]
    mat = np.zeros((len(contrasts), len(feats)))
    for i, name in enumerate(contrasts):
        sub = effects[effects.contrast == name]
        for j, f in enumerate(feats):
            hit = sub[sub.feature == f]
            mat[i, j] = float(hit["cohens_d"].iloc[0]) if len(hit) else np.nan
    fig, ax = plt.subplots(figsize=(11.0, 4.4), layout="constrained")
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=40, ha="right")
    ax.set_yticks(range(len(contrasts)))
    ax.set_yticklabels(contrasts)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Cohen's d (A − B)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=None, help="Cap takes per class (None = all).")
    parser.add_argument("--train-per-class", type=int, default=40, help="Cap for training only.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse", action="store_true", help="Reuse results/raw_features.pkl if present.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "raw_features.pkl"
    spec_path = OUT_DIR / "mean_spectra.npz"

    if args.reuse and cache_path.exists() and spec_path.exists():
        df = pd.read_pickle(cache_path)
        spec_npz = np.load(spec_path)
        spec_map = {k: spec_npz[k] for k in ORDER}
        print(f"reused {cache_path}  n={len(df)}")
    else:
        frames = []
        spec_map = {}
        for ds in DATASETS:
            cap = args.train_per_class if ds["id"] == "training" else args.per_class
            print(f"extracting {ds['id']}  per_class={cap}  workers={args.workers}", flush=True)
            frame, spec = extract_dataset(ds, cap, args.seed, args.workers)
            print(
                f"  n={len(frame)} ok={int(frame.ok.sum())} silent={int(frame.silent.fillna(False).sum())}",
                flush=True,
            )
            frames.append(frame)
            spec_map[ds["id"]] = spec
        df = pd.concat(frames, ignore_index=True)
        df.to_pickle(cache_path)
        df.to_csv(OUT_DIR / "raw_features.csv", index=False)
        np.savez(spec_path, **spec_map, freqs=np.fft.rfftfreq(N_FFT, 1.0 / SR_TARGET))
        print(f"wrote {cache_path}")

    summary = summarize(df)
    summary.to_csv(OUT_DIR / "summary_means.csv", index=False)

    effect_cols = [
        "rms_db",
        "peak_db",
        "crest",
        "zcr",
        "centroid_hz",
        "bandwidth_hz",
        "rolloff_hz",
        "log_flatness",
        "slope",
        "hf_ratio",
        "hf_db",
        "attack_s",
        "decay_s",
        "onset_s",
        "temporal_centroid",
        "first_20ms_db",
        "noise_floor_db",
        "snr_db",
        "silence_ratio",
        "ch_corr",
        "mid_side_db",
        "band_rumble",
        "band_low",
        "band_fund",
        "band_mid",
        "band_presence",
        "band_air",
        "band_mid_log",
        "band_presence_log",
        "band_air_log",
        "mfcc_1",
        "mfcc_2",
        "mfcc_3",
    ]
    effects = contrast_table(df, effect_cols)
    effects.to_csv(OUT_DIR / "contrasts_cohens_d.csv", index=False)

    # Drop stereo-only columns that are NaN on training from the classifier set.
    timbre = [c for c in TIMBRE_COLS if c in df.columns]
    held = df[df.dataset != "training"]
    acc_all = classify(df, timbre, "dataset")
    acc_kb = classify(held, timbre, "keyboard")
    acc_mic = classify(held, timbre, "mic")
    acc = pd.concat([acc_all, acc_kb, acc_mic], ignore_index=True)
    # Fix the keyboard/mic rows labels (classify uses the y column name).
    pair = pairwise_acc(df, timbre)
    acc.to_csv(OUT_DIR / "classifier_multiclass.csv", index=False)
    pair.to_csv(OUT_DIR / "classifier_pairwise.csv", index=False)

    top = (
        effects.sort_values(["contrast", "abs_d"], ascending=[True, False])
        .groupby("contrast", sort=False)
        .head(6)
    )

    boxplot_panel(
        df,
        [
            ("rms_db", "RMS (dB)", -35, -12),
            ("centroid_hz", "STFT centroid (Hz)", 200, 1400),
            ("band_low", "Energy 80–250 Hz", 0, 0.7),
            ("band_mid", "Energy 0.8–2.5 kHz", 0, 0.5),
            ("log_flatness", "log10 spectral flatness", None, None),
            ("onset_s", "Onset delay (s)", 0, 0.25),
            ("decay_s", "Decay to 1/e (s)", 0, 1.6),
            ("first_20ms_db", "First 20 ms RMS (dB)", -90, -10),
        ],
        OUT_DIR / "boxplots.png",
    )
    plot_spectra(spec_map, OUT_DIR / "mean_spectra.png")
    plot_pca(df, timbre, OUT_DIR / "pca_timbre.png")
    plot_bands(df, OUT_DIR / "band_energy.png")
    plot_effect_heatmap(effects, OUT_DIR / "cohens_d.png")

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    print("\n=== inventory / means ===")
    show = [
        "label",
        "keyboard",
        "mic",
        "n",
        "n_silent",
        "n_channels",
        "duration_s",
        "rms_db",
        "centroid_hz",
        "slope",
        "hf_db",
        "log_flatness",
        "onset_s",
        "decay_s",
        "first_20ms_db",
        "ch_corr",
        "mid_side_db",
        "band_low",
        "band_mid",
        "mfcc_1",
    ]
    print(summary[show].to_string(index=False))

    print("\n=== largest |d| per contrast ===")
    print(
        top[["contrast", "feature", "mean_a", "mean_b", "delta", "cohens_d"]]
        .to_string(index=False)
    )

    print("\n=== 5-way / factor classifiers (level-invariant timbre features) ===")
    print(acc.to_string(index=False))
    print("\n=== pairwise dataset ID from timbre ===")
    print(pair.to_string(index=False))

    # Quality × keyboard interaction on centroid / presence.
    ok = df[df.ok & ~df.silent.fillna(False)]
    print("\n=== centroid (Hz) by quality ===")
    print(
        ok.pivot_table(index="dataset", columns="quality", values="centroid_hz", aggfunc="mean")
        .reindex(ORDER)
        .to_string()
    )
    print("\n=== presence band by quality ===")
    print(
        ok.pivot_table(index="dataset", columns="quality", values="band_presence", aggfunc="mean")
        .reindex(ORDER)
        .to_string()
    )


if __name__ == "__main__":
    main()
