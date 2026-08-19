#!/usr/bin/env python3
"""Score each cnn-latest checkpoint on its own Techno test split, clean and
under the same eval-only overlays used on the recorded OOD sets.

This is the in-domain control: no recorded device/keyboard change, only
synthetic noise / RIR / DIR. Writes results/indomain_overlay_seeds.csv.
"""
import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from scipy.signal import fftconvolve

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent
FEATURES_DIR = ROOT / "features"
AUDIO_DIR = ROOT / "datasets" / "training"
WEIGHTS_DIR = HERE / "weights"
SPLITS_DIR = HERE / "results" / "splits"
OUT_CSV = HERE / "results" / "indomain_overlay_seeds.csv"

VARIANTS = ["orig", "clean"]
SEEDS = [42, 43, 44, 45, 46]
VARIANT_NPZ = {
    "orig": FEATURES_DIR / "training.npz",
    "clean": FEATURES_DIR / "training-clean.npz",
}
NOISE_BANK = FEATURES_DIR / "noise" / "esc50-interior_domestic.npz"
RIR_BANK = FEATURES_DIR / "rir" / "ir-survey-indoor_no_bathroom.npz"
DIR_BANK = FEATURES_DIR / "dir" / "dirs-micirp.npz"

SNR_DB_RANGE = (10.0, 25.0)
EPS = 1e-10
SR = 48_000
HOP = 512
N_BINS = 216
BPO = 36
FRAMES = 188
FMIN = librosa.note_to_hz("C1")
BATCH_SIZE = 32

# Independent streams so overlay draws do not share a sequence.
RNG_OFFSET = {"noise": 1000, "rir": 2000, "dir": 3000}


def db_to_amp(x_db):
    return np.power(10.0, x_db.astype(np.float64) / 20.0)


def amp_to_db(x_amp):
    return (20.0 * np.log10(np.maximum(x_amp, EPS))).astype(np.float32)


def mix_power(chord_db, noise_db, snr_db, roll=0):
    chord_amp = db_to_amp(chord_db)
    noise_amp = db_to_amp(noise_db)
    if roll:
        noise_amp = np.roll(noise_amp, int(roll), axis=1)
    e_c = float(np.mean(np.square(chord_amp)) + EPS)
    e_n = float(np.mean(np.square(noise_amp)) + EPS)
    scale = np.sqrt(e_c / (e_n * (10.0 ** (snr_db / 10.0))))
    mixed = np.sqrt(np.square(chord_amp) + np.square(scale * noise_amp))
    return amp_to_db(mixed)


def extract_cqt(y):
    cqt = np.abs(librosa.cqt(
        y=y, sr=SR, fmin=FMIN, n_bins=N_BINS,
        bins_per_octave=BPO, hop_length=HOP, window="hann",
    ))
    cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    if cqt.shape[1] < FRAMES:
        cqt = np.pad(cqt, ((0, 0), (0, FRAMES - cqt.shape[1])))
    else:
        cqt = cqt[:, :FRAMES]
    return cqt.astype(np.float32)


def convolve_crop(y, h):
    wet = fftconvolve(y, h, mode="full")[: len(y)]
    peak = float(np.max(np.abs(wet)))
    if peak > 0:
        wet = wet / peak
    return wet.astype(np.float32)


def class_order():
    labels = np.load(VARIANT_NPZ["orig"], allow_pickle=True)["labels"].astype(str)
    return np.array(sorted(np.unique(labels)))


def already_done(variant, seed, overlay):
    if not OUT_CSV.exists():
        return False
    prev = pd.read_csv(OUT_CSV)
    return bool(((prev.variant == variant) & (prev.seed == seed) & (prev.overlay == overlay)).any())


def append_row(row):
    prev = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()
    prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    prev.to_csv(OUT_CSV, index=False)


def score(model, X, y):
    from sklearn.metrics import precision_recall_fscore_support

    if X.ndim == 3:
        X = X[..., None]
    pred = np.argmax(model.predict(X, verbose=0, batch_size=BATCH_SIZE), axis=1)
    acc = float((pred == y).mean())
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, labels=np.arange(36), average="macro", zero_division=0
    )
    return acc, float(p), float(r), float(f1)


def mix_noise(X, rng, noise_x):
    out = np.empty_like(X)
    n_pool = len(noise_x)
    frames = X.shape[-1]
    for i in range(len(X)):
        snr = float(rng.uniform(*SNR_DB_RANGE))
        n_i = int(rng.integers(0, n_pool))
        roll = int(rng.integers(0, frames))
        out[i] = mix_power(X[i], noise_x[n_i], snr, roll=roll)
    return out


def mix_ir(files, rng, irs):
    out = np.empty((len(files), N_BINS, FRAMES), dtype=np.float32)
    n_ir = len(irs)
    for i, rel in enumerate(files):
        h = np.asarray(irs[int(rng.integers(0, n_ir))], dtype=np.float32)
        path = AUDIO_DIR / rel
        y, _ = librosa.load(path, sr=SR, mono=True)
        out[i] = extract_cqt(convolve_crop(y.astype(np.float32), h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import LabelEncoder

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", tf.config.experimental.get_device_details(gpus[0]).get("device_name"))

    class_names = class_order()
    le = LabelEncoder().fit(class_names)
    noise_x = np.load(NOISE_BANK, allow_pickle=True)["features"].astype(np.float32)
    rir_irs = np.load(RIR_BANK, allow_pickle=True)["irs"]
    dir_irs = np.load(DIR_BANK, allow_pickle=True)["irs"]

    variants = [args.variant] if args.variant else VARIANTS
    seeds = [args.seed] if args.seed is not None else SEEDS

    for variant in variants:
        d = np.load(VARIANT_NPZ[variant], allow_pickle=True)
        features = d["features"].astype(np.float32)
        labels = d["labels"].astype(str)
        files_npz = d["files"].astype(str) if "files" in d.files else None

        for seed in seeds:
            wp = WEIGHTS_DIR / f"{variant}-seed{seed}.keras"
            split_path = SPLITS_DIR / f"{variant}-seed{seed}.csv"
            if not wp.exists() or not split_path.exists():
                print(f"missing {wp.name} or split, skip")
                continue
            split = pd.read_csv(split_path)
            test = split[split.split == "test"].sort_values("source_index")
            idx = test["source_index"].to_numpy()
            X = features[idx]
            y = le.transform(labels[idx])
            files = (
                files_npz[idx] if files_npz is not None
                else test["file"].astype(str).to_numpy()
            )

            need = [ov for ov in ("none", "noise", "rir", "dir") if not already_done(variant, seed, ov)]
            if not need:
                print(f"skip {variant}/seed{seed} (all overlays done)")
                continue

            model = load_model(wp)
            for overlay in need:
                if overlay == "none":
                    Xx = X
                    factor = "none"
                elif overlay == "noise":
                    rng = np.random.default_rng(seed + RNG_OFFSET["noise"])
                    Xx = mix_noise(X, rng, noise_x)
                    factor = "environment"
                elif overlay == "rir":
                    rng = np.random.default_rng(seed + RNG_OFFSET["rir"])
                    print(f"  {variant}/seed{seed} RIR on {len(files)} clips...", flush=True)
                    Xx = mix_ir(files, rng, rir_irs)
                    factor = "environment"
                else:
                    rng = np.random.default_rng(seed + RNG_OFFSET["dir"])
                    print(f"  {variant}/seed{seed} DIR on {len(files)} clips...", flush=True)
                    Xx = mix_ir(files, rng, dir_irs)
                    factor = "recording-setting"

                acc, p, r, f1 = score(model, Xx, y)
                recorded = None
                if overlay != "none" and OUT_CSV.exists():
                    prev = pd.read_csv(OUT_CSV)
                    hit = prev[(prev.variant == variant) & (prev.seed == seed) & (prev.overlay == "none")]
                    if len(hit):
                        recorded = float(hit.iloc[0]["accuracy"])
                tax = (recorded - acc) if recorded is not None else 0.0
                row = {
                    "variant": variant,
                    "seed": seed,
                    "dataset": "training-test",
                    "overlay": overlay,
                    "factor": factor,
                    "n": int(len(idx)),
                    "accuracy": acc,
                    "macro_precision": p,
                    "macro_recall": r,
                    "macro_f1": f1,
                    "tax": tax,
                }
                append_row(row)
                print(f"OK {variant}/seed{seed}/{overlay}: acc={acc:.4f} tax={tax:.4f} n={len(idx)}")
                if overlay != "none":
                    del Xx

            del model
            tf.keras.backend.clear_session()

    return 0


if __name__ == "__main__":
    sys.exit(main())
