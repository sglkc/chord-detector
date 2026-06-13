# Real-Time Chord Detection Demo

A small full-stack demo of the chord-detection research in this
repository. The browser captures the microphone, ships raw 48 kHz
mono Float32 PCM to a Python backend over a single WebSocket, and the
backend runs the **same** librosa CQT + Superflux onset + CNN
pipeline used in the offline notebooks (`EVALUATION.md`). The
spectrogram and the latest predicted chord stream back to the page in
real time.

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
                                              ChordClassifier (TF / Keras CNN)
                                                  │
                              ◀──── JSON ────────┘
                              {type: "cqt_columns"}  {type: "chord"}
```

## Run

```bash
# from the project root
uv sync                                  # install deps (includes fastapi/uvicorn)
uv run uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
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
├── pipeline/
│   ├── __init__.py
│   ├── audio_buffer.py        # bounded Float32 ring buffer
│   ├── cqt_stream.py          # incremental librosa CQT
│   ├── onset_detector.py      # Superflux envelope + peak pick
│   ├── segment_buffer.py      # state machine: 188-frame windows
│   ├── classifier.py          # TF/Keras CNN + linear-interp stretch
│   └── runner.py              # per-connection orchestrator
├── templates/
│   └── index.html             # single-page shell
└── static/
    ├── styles.css
    ├── app.js                 # main script (transport + canvas)
    └── worklet.js             # AudioWorklet source (also inlined in app.js)
```

## WebSocket protocol

Audio: client → server **binary** frames. Each frame is a
little-endian Float32Array of mono samples. The recommended chunk size
is 4096 samples (≈ 85 ms @ 48 kHz).

Server → client **text** JSON frames:

| type           | fields                                                                                |
| -------------- | ------------------------------------------------------------------------------------- |
| `ready`        | Emitted on connect.                                                                   |
| `cqt_columns`  | `n_bins, n_cols, time_s, columns[Array<number>]` – flat dB-scaled CQT in row-major.   |
| `chord`        | `raw_label, display_label, confidence, predicted_index, onset_time, duration, truncated, source_frames`. |
| `error`        | `message`                                                                             |
| `pong`         | Reply to a client `ping` text frame.                                                  |

Client → server **text** frames (control):
`ping`, `reset`.

## Pipeline details

* **Sample rate**: 48 kHz mono (matches the training data).
* **CQT**: 216 bins (6 octaves × 36 bins/octave), fmin = C1, hop = 512.
* **Onset detector**: Superflux (lag=2, max_size=3) + the same peak-pick
  parameters as
  `notebooks/onset/onset-classify_superflux.ipynb`. A minimum onset
  gap of 80 ms is enforced in the `SegmentBuffer` to debounce
  vibrato.
* **Segment window**: 188 frames (≈ 2 s of audio). Anything shorter
  is **linearly interpolated** to 188 frames before being fed to the
  CNN (this is the same `np.interp` stretch used in the offline
  training pipeline).
* **Classifier**: `models/model-bn-c64-c128-c256-c256-d256.keras`
  (loaded once per WebSocket connection). Output is argmax-decoded
  with a short display string (e.g. `C:maj`).

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
| `SEND_CQT_EVERY_MS` | 50 | throttles CQT updates to the client |
| `CQT_TRAIL_COLUMNS` | 80 | how many trailing CQT cols are sent per update |
| `WS_CHUNK_SAMPLES` | 4096 | browser-side audio chunk size |

## Limitations

* **Latency**: ~85 ms per network hop + ~30 ms for librosa CQT on
  CPU + ~10 ms for CNN inference. End-to-end you should see a chord
  update within ~150–200 ms of a clear attack.
* **Onset false-positives**: Superflux can fire on percussive attacks
  (drum hits, transients). The CNN will still try to classify the
  window, often producing an unexpected chord.
* **No silence gate**: in a silent room the spectrogram will still
  scroll; it just won't produce chord messages.
* **First 2 s are ignored**: CQT / onset detection require the
  minimum training window to fill.

## Troubleshooting

| symptom | fix |
| ------- | --- |
| Browser says "NotAllowedError" | grant mic permission in the address bar |
| `WebSocket closed before connection established` | is the server running? try `curl http://localhost:8000/healthz` |
| CQT canvas never updates | the mic might be muted, or the sample rate is not 48 kHz. Open devtools and check the WS frames |
| `ModuleNotFoundError: No module named 'fastapi'` | run `uv sync` to install the new deps |
| `OSError: Unable to open file` for the .keras model | confirm `models/model-bn-c64-c128-c256-c256-d256.keras` exists |
