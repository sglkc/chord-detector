# Real-Time Chord Detection Demo

A small full-stack demo of the chord-detection research in this
repository. The browser captures the microphone, ships raw 48 kHz
mono Float32 PCM to a Python backend over a single WebSocket, and the
backend runs the **same** librosa CQT + Superflux onset + CNN
pipeline used in the offline notebooks (`EVALUATION.md`). The
spectrogram and the latest predicted chord stream back to the page in
real time. The browser also keeps the captured take and a scrollable
list of every closed segment; after Stop, Play seeks that recording.

```
   Browser                                       Python (uvicorn :8000)
   ───────                                       ──────────────────────
   AudioContext @ 48 kHz  ───── Float32 ──────▶  /ws
   AudioWorklet              WebSocket          │
   getUserMedia                                  ▼
                                              AudioRingBuffer
                                                  │
                                                  ▼
                                              CQTStream  (librosa.cqt, 216 bins)
                                                  │
                                                  ▼
                                              OnsetDetector (Superflux + peak pick)
                                                  │
                                                  ▼
                                              SegmentBuffer  (188-frame windows)
                                                  │
                                                  ▼
                                              ChordClassifier (shared TF / Keras CNN)
                                                  │
                              ◀──── JSON ────────┘
                              {type: "cqt_columns"}  {type: "onset"}  {type: "chord"}
```

## Run

```bash
# from the repository root
uv sync                                  # install deps (includes fastapi/uvicorn)
# one-time: link the trained weights (gitignored; canonical file stays under training/)
mkdir -p app/models
ln -sfn ../../training/models/model-bn-c64-c128-c256-c256-d256.keras \
  app/models/model-bn-c64-c128-c256-c256-d256.keras
uv run uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

## Tests

```bash
# pipeline unit tests (no server, no TensorFlow model load required for most cases)
.venv/bin/python -m app.tests.test_pipeline_components

# process-level e2e: boots uvicorn, checks /healthz, static assets, WebSocket
# CQT stream, and (if present) classification on a local validation clip
.venv/bin/python -m app.tests.test_e2e_server
```

…then open <http://localhost:8000> in a browser. The browser will
prompt for microphone access.

> **HTTPS note**: `getUserMedia` requires HTTPS or `localhost`. To use
> a remote machine, front the app with an HTTPS reverse proxy (Caddy,
> nginx, etc.) and serve over `https://`.

## Layout

```
app/
├── __init__.py
├── config.py                  # shared constants (CQT, onsets, model)
├── server.py                  # FastAPI app + WebSocket endpoint
├── README.md
├── models/                    # gitignored symlink(s) → training/models/
│   └── model-bn-….keras
├── pipeline/
│   ├── __init__.py
│   ├── audio_buffer.py        # bounded Float32 ring buffer
│   ├── cqt_stream.py          # incremental librosa CQT
│   ├── onset_detector.py      # Superflux envelope + peak picking
│   ├── segment_buffer.py      # state machine: 188-frame windows
│   ├── classifier.py          # TF/Keras CNN + linear-interp stretch
│   └── runner.py              # per-connection orchestrator
├── templates/
│   └── index.html             # single-page shell
└── static/
    ├── styles.css
    ├── app.js                 # main script (transport + canvas)
    └── worklet.js             # AudioWorklet (loaded via addModule)
```

## WebSocket protocol

Audio: client → server **binary** frames. Each frame is a
little-endian Float32Array of mono samples. The recommended chunk size
is 4096 samples (≈ 85 ms @ 48 kHz). Frames larger than
`WS_CHUNK_SAMPLES * 4 * 4` bytes are rejected.

Server → client **text** JSON frames:

| type           | fields                                                                                |
| -------------- | ------------------------------------------------------------------------------------- |
| `ready`        | Emitted on connect.                                                                   |
| `model_status` | `loaded`, `load_time_s`, `error` – startup model load result.                         |
| `cqt_columns`  | `n_bins, n_cols, time_s, columns[Array<number>]` – full trail snapshot, C-order flat. |
| `onset`        | `column, time_s, strength` – Superflux onset at a global CQT column (spectrogram marker). |
| `chord`        | `raw_label, display_label, confidence, predicted_index, onset_time, duration, truncated, source_frames, onset_column, strength`. |
| `error`        | `message`                                                                             |
| `pong`         | Reply to a client `ping` text frame.                                                  |

`cqt_columns.columns` is a **full trail snapshot** of the last
`CQT_TRAIL_COLUMNS` (not a delta). Layout is C-order
`(n_bins, n_cols)` so index `b * n_cols + c`. The client rebuilds the
spectrogram display from each snapshot.

Client → server **text** frames (control):
`ping`, `reset`, `set <key>=<value>`.

## Pipeline details

* **Sample rate**: 48 kHz mono (matches the training data). The
  browser refuses to stream if `AudioContext.sampleRate !== 48000`.
* **CQT**: 216 bins (6 octaves × 36 bins/octave), fmin = C1, hop = 512.
  Streaming dB conversion uses a **fixed** `ref=1.0` so stitched
  columns stay comparable across analysis windows. The CNN input is
  then peak-shifted to 0 dB (`ref=np.max`) to match training.
* **Onset detector**: Superflux (lag=2, max_size=3) run on a trailing
  CQT **context** window (not tiny per-tick slices alone), with the
  same peak-pick parameters as
  `notebooks/onset/onset-classify_superflux.ipynb`. Envelope history
  is capped. A minimum onset gap of 80 ms is enforced in the
  `SegmentBuffer` to debounce vibrato.
* **Segment window**: 188 frames (≈ 2 s of audio). Anything shorter
  is **linearly interpolated** to 188 frames before being fed to the
  CNN (this is the same `np.interp` stretch used in the offline
  training pipeline). After a truncate-on-new-onset, collection
  restarts immediately at the new onset.
* **Classifier**: `app/models/model-bn-c64-c128-c256-c256-d256.keras`
  (symlink to `training/models/…`), loaded **once at app startup**
  (FastAPI lifespan) and shared across WebSocket connections.
  Concurrent `predict` calls are serialized with a lock. Per-connection
  state is only the DSP pipeline (buffer / CQT / onsets / segments).

## Latency expectations

Two different latencies are easy to confuse:

1. **Spectrogram update latency** – the scrolling CQT canvas updates
   about every `SEND_CQT_EVERY_MS` (default 50 ms) once ~2 s of audio
   have buffered for a stable CQT. This is *not* a chord label.
2. **First chord label latency** – a classification is emitted only
   when a segment window completes:
   * **Full window**: after an onset, once **188 frames**
     (≈ 2.0 s @ hop 512 / 48 kHz) have been collected; or
   * **Truncated window**: earlier, if a **new onset** arrives before
     188 frames fill (the partial window is stretched and classified).

So the first chord often appears only after ~2 s of post-onset audio,
or sooner on a rapid second attack. DSP + network per tick is usually
on the order of tens to a couple of hundred milliseconds once the
window is ready; that path latency is separate from the window-fill
time above.

The shared model is loaded during server startup (see `/healthz`
`model_loaded` / `model_load_time_s`), not on each WebSocket connect.

## Tunables

All constants live in `app/config.py`:

| name | default | effect |
| ---- | ------- | ------ |
| `AUDIO_SAMPLE_RATE` | 48000 | must match training |
| `CQT_HOP_LENGTH` | 512 | smaller = better time resolution, more CPU |
| `CQT_OCTAVES` / `CQT_BINS_PER_OCTAVE` | 6 / 36 | must match the model |
| `CQT_FEATURE_FRAMES` | 188 | must match the model |
| `PEAK_PICK_PARAMETERS` | (see config) | more `wait` = fewer onsets |
| `MIN_ONSET_GAP_MS` | 80 | debounce vibrato |
| `ONSET_WARMUP_FRAMES` | 188 | ignore early spurious onsets |
| `SEND_CQT_EVERY_MS` | 50 | throttles CQT updates to the client |
| `CQT_TRAIL_COLUMNS` | 80 | how many trailing CQT cols are sent per update |
| `WS_CHUNK_SAMPLES` | 4096 | browser-side audio chunk size |

## Limitations

* **Chord label lag vs spectrogram**: see *Latency expectations*
  above — filling a 188-frame window dominates first-label delay.
* **Onset false-positives**: Superflux can fire on percussive attacks
  (drum hits, transients). The CNN will still try to classify the
  window, often producing an unexpected chord.
* **No silence class**: the CNN is 36 closed-set chords (16 silent
  training clips were labeled `C_diminished_4`).
* **First 2 s of audio**: CQT / onset detection require the minimum
  training window to fill before any columns are produced.

## Troubleshooting

| symptom | fix |
| ------- | --- |
| Browser says "NotAllowedError" | grant mic permission in the address bar |
| `WebSocket closed before connection established` | is the server running? try `curl http://localhost:8000/healthz` |
| Status shows sample-rate error | browser would not open AudioContext at 48 kHz; use a browser/device that supports it |
| CQT canvas never updates | the mic might be muted, or check WS frames in devtools |
| `model_loaded: false` on `/healthz` | recreate the `app/models/*.keras` symlink; confirm `training/models/` has the file; check server logs |
| `ModuleNotFoundError: No module named 'fastapi'` | run `uv sync` from the repo root |
| `OSError: Unable to open file` for the .keras model | symlink missing/broken, or weights missing under `training/models/` |
