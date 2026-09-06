# processed-samples

Feature-faithful audio exports for listening to what the augmentation pipelines feed the model.

## Layout

```
processed-samples/
  index.csv
  README.md
  vivo/
  thinkpad/
  flow-old/
    {index}__{chord_path}__dry.wav
    {index}__{chord_path}__rir.wav
    {index}__{chord_path}__dir.wav
    {index}__{chord_path}__clean_cqt_inv.wav
    {index}__{chord_path}__noise_cqt_inv.wav
```

## Conditions (do not human-sweeten)

| Suffix | Source of truth |
|--------|-----------------|
| `dry` | Mono 48 kHz load of the source clip |
| `rir` | Exact `mix-rir-cqt`: convolve with manifest RIR → crop → peak-norm |
| `dir` | Exact `mix-dir-cqt`: convolve with manifest DIR → crop → peak-norm |
| `clean_cqt_inv` | Griffin–Lim inverse of the **stored clean CQT** feature matrix |
| `noise_cqt_inv` | Griffin–Lim inverse of the **stored noise-mixed CQT** feature matrix |

Noise features are **power-domain CQT mixes**, not time-domain `x+n`. Inversion is approximate (phase not stored) but uses the **same matrices** the CNN evaluates. RIR/DIR wavs are the exact pre-CQT wet waveform.

## How to regenerate

Run in order:

1. `training/notebooks/aug-dir/dir-prep.ipynb` (and RIR/noise prep if missing)
2. `training/notebooks/aug-dir/mix-dir-cqt.ipynb`
3. `training/notebooks/export-processed-samples.ipynb`

Config: `N_PER_DOMAIN`, `SEED`, `FORCE_REEXPORT`.

## Listening tips

1. A/B **dry vs rir** (room), **dry vs dir** (mic color) — time-domain, bit-faithful.
2. A/B **clean_cqt_inv vs noise_cqt_inv** (feature-domain; both inverted the same way).
3. Do not expect `noise_cqt_inv` to sound like a simple overdub of ESC-50 under the chord — that is not what the model sees.
