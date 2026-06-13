"""Shared configuration for the real-time chord detection demo.

All constants are copied verbatim from the offline training / evaluation
notebooks (``notebooks/onset/onset-classify_superflux.ipynb`` and
``show_cqt.py``) so the live pipeline stays in lock-step with the
trained CNN. If you change a value here you almost certainly need to
retrain the model.

The constants are grouped into four sections:

* **Audio / CQT**    - sample rate, hop length, frequency resolution.
* **Onset detection** - Superflux and peak-picking parameters.
* **Model**         - path, label list, expected input frame count.
* **Networking**    - WebSocket chunk size and rate-limits.
"""

from __future__ import annotations

from pathlib import Path

import librosa


# ---------------------------------------------------------------------------
# Audio + CQT
# ---------------------------------------------------------------------------

#: Target sample rate (Hz). Matches the training data exactly.
AUDIO_SAMPLE_RATE: int = 48_000

#: Minimum window of audio (seconds) before CQT / onset detection runs.
#: Equal to ``AUDIO_SAMPLE_RATE * AUDIO_MIN_SECONDS / CQT_HOP_LENGTH``
#: which yields 188 frames in the superflux notebook.
AUDIO_MIN_SECONDS: float = 2.0

#: STFT / CQT hop length (samples). 512 @ 48 kHz -> ~10.7 ms per frame.
CQT_HOP_LENGTH: int = 512

#: Number of octaves the CQT spans.
CQT_OCTAVES: int = 6

#: Frequency bins per octave (36 = 3 bins per semitone).
CQT_BINS_PER_OCTAVE: int = 36

#: Total number of CQT frequency bins (octaves * bins_per_octave).
CQT_FEATURE_BINS: int = CQT_BINS_PER_OCTAVE * CQT_OCTAVES  # 216

#: Lowest note of the CQT (C1, ~32.7 Hz).
CQT_FMIN: float = float(librosa.note_to_hz("C1"))

#: Number of CQT frames fed to the CNN. Matches the training pipeline.
#: ``round(2 * 48000 / 512) = 188``.
CQT_FEATURE_FRAMES: int = 188

#: Maximum audio kept in the rolling buffer (seconds). Old audio is
#: discarded so the CQT / onset stages never see more than this.
MAX_AUDIO_SECONDS: float = 6.0


# ---------------------------------------------------------------------------
# Onset detection (Superflux, Boeck & Widmer 2013)
# ---------------------------------------------------------------------------

#: Superflux parameters. ``lag=2`` and ``max_size=3`` are the values
#: from the original paper and from
#: ``notebooks/onset/onset-classify_superflux.ipynb``.
SUPERFLUX_PARAMETERS: dict = {
    "lag": 2,
    "max_size": 3,
}

#: Peak-picking parameters used by ``librosa.onset.onset_detect``.
#: Same numbers as the training notebook.
PEAK_PICK_PARAMETERS: dict = {
    "pre_max": 30,
    "post_max": 1,
    "pre_avg": 100,
    "post_avg": 1,
    "wait": 30,
}

#: Minimum gap between two consecutive onsets (ms). Acts as a debounce
#: against Superflux double-triggers on vibrato. The algorithm itself
#: does not enforce this; ``SegmentBuffer`` honors it.
MIN_ONSET_GAP_MS: int = 80

#: Onsets detected in the first ``ONSET_WARMUP_FRAMES`` of envelope
#: history are ignored. They are typically the artifact of computing
#: the spectral flux on a new signal that starts in silence / noise.
#: One CQT window's worth of frames matches the offline pipeline.
ONSET_WARMUP_FRAMES: int = CQT_FEATURE_FRAMES


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

#: Filesystem path to the trained Keras CNN. Resolved relative to the
#: project root (``training/``).
MODEL_PATH: Path = (
    Path(__file__).resolve().parent.parent / "models" / "model-bn-c64-c128-c256-c256-d256.keras"
)

#: Class label order, indexed by ``argmax`` of the CNN output.
#: Order MUST match the model. 36 classes = 12 roots * 3 qualities.
MODEL_LABELS: list = [
    "A#_diminished_4", "A#_major_4", "A#_minor_4",
    "A_diminished_4", "A_major_4", "A_minor_4",
    "B_diminished_4", "B_major_4", "B_minor_4",
    "C#_diminished_4", "C#_major_4", "C#_minor_4",
    "C_diminished_4", "C_major_4", "C_minor_4",
    "D#_diminished_4", "D#_major_4", "D#_minor_4",
    "D_diminished_4", "D_major_4", "D_minor_4",
    "E_diminished_4", "E_major_4", "E_minor_4",
    "F#_diminished_4", "F#_major_4", "F#_minor_4",
    "F_diminished_4", "F_major_4", "F_minor_4",
    "G#_diminished_4", "G#_major_4", "G#_minor_4",
    "G_diminished_4", "G_major_4", "G_minor_4",
]

#: Map from raw model label (e.g. ``"C_major_4"``) to a short display
#: form (e.g. ``"C:maj"``). Used by the WebSocket payload so the
#: browser does not need a 36-entry lookup table.
LABEL_DISPLAY_MAP: dict = {
    "A#_diminished_4": "A#:dim", "A#_major_4": "A#:maj", "A#_minor_4": "A#:min",
    "A_diminished_4": "A:dim",  "A_major_4": "A:maj",   "A_minor_4": "A:min",
    "B_diminished_4": "B:dim",  "B_major_4": "B:maj",   "B_minor_4": "B:min",
    "C#_diminished_4": "C#:dim", "C#_major_4": "C#:maj", "C#_minor_4": "C#:min",
    "C_diminished_4": "C:dim",  "C_major_4": "C:maj",   "C_minor_4": "C:min",
    "D#_diminished_4": "D#:dim", "D#_major_4": "D#:maj", "D#_minor_4": "D#:min",
    "D_diminished_4": "D:dim",  "D_major_4": "D:maj",   "D_minor_4": "D:min",
    "E_diminished_4": "E:dim",  "E_major_4": "E:maj",   "E_minor_4": "E:min",
    "F#_diminished_4": "F#:dim", "F#_major_4": "F#:maj", "F#_minor_4": "F#:min",
    "F_diminished_4": "F:dim",  "F_major_4": "F:maj",   "F_minor_4": "F:min",
    "G#_diminished_4": "G#:dim", "G#_major_4": "G#:maj", "G#_minor_4": "G#:min",
    "G_diminished_4": "G:dim",  "G_major_4": "G:maj",   "G_minor_4": "G:min",
}


# ---------------------------------------------------------------------------
# Networking / WebSocket
# ---------------------------------------------------------------------------

#: How many audio samples the browser sends per WebSocket frame.
#: 4096 samples @ 48 kHz ~= 85 ms of audio per frame.
WS_CHUNK_SAMPLES: int = 4096

#: Minimum interval between CQT column-update messages (ms). The
#: pipeline still computes CQT every tick; this only rate-limits the
#: outbound JSON payload. 50 ms ~= 20 fps, smooth for a scrolling
#: canvas without saturating the WebSocket.
SEND_CQT_EVERY_MS: int = 50

#: How many trailing CQT columns the server includes in each
#: ``cqt_columns`` message. ``80`` covers ~0.85 s of recent audio
#: which gives the canvas ~600 px of horizontal history.
CQT_TRAIL_COLUMNS: int = 80


# ---------------------------------------------------------------------------
# Paths (computed)
# ---------------------------------------------------------------------------

#: Directory that holds the static frontend assets.
STATIC_DIR: Path = Path(__file__).resolve().parent / "static"

#: Directory that holds Jinja-free HTML templates.
TEMPLATES_DIR: Path = Path(__file__).resolve().parent / "templates"


__all__ = [
    "AUDIO_SAMPLE_RATE",
    "AUDIO_MIN_SECONDS",
    "CQT_HOP_LENGTH",
    "CQT_OCTAVES",
    "CQT_BINS_PER_OCTAVE",
    "CQT_FEATURE_BINS",
    "CQT_FMIN",
    "CQT_FEATURE_FRAMES",
    "MAX_AUDIO_SECONDS",
    "SUPERFLUX_PARAMETERS",
    "PEAK_PICK_PARAMETERS",
    "MIN_ONSET_GAP_MS",
    "MODEL_PATH",
    "MODEL_LABELS",
    "LABEL_DISPLAY_MAP",
    "WS_CHUNK_SAMPLES",
    "SEND_CQT_EVERY_MS",
    "CQT_TRAIL_COLUMNS",
    "STATIC_DIR",
    "TEMPLATES_DIR",
]
