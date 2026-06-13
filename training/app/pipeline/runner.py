"""Top-level pipeline orchestrator.

One :class:`PipelineRunner` is created per WebSocket connection. It
owns the audio buffer, CQT stream, onset detector, segment buffer and
classifier, and exposes a single :py:meth:`ingest_pcm` method that the
WebSocket handler calls whenever a new audio chunk arrives.

The runner returns a list of *messages* to ship back to the browser.
The two message types are:

``{"type": "cqt_columns", "columns": [...], "n_cols": int, "time_s": float}``

    A 1-D flat array of ``n_bins * n_cols`` floats representing the
    trailing CQT columns (dB). The browser paints them right-to-left
    on a scrolling canvas.

``{"type": "chord", "raw_label": str, "display_label": str,
   "confidence": float, "onset_time": float, "duration": float,
   "truncated": bool, "source_frames": int}``

    The result of a CNN inference on a completed segment.

The runner is intentionally synchronous (no asyncio) - the WebSocket
handler is expected to call it inside ``asyncio.to_thread``.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np

from .. import config
from .audio_buffer import AudioRingBuffer
from .classifier import ChordClassifier
from .cqt_stream import CQTStream
from .onset_detector import OnsetDetector
from .segment_buffer import SegmentBuffer


class PipelineRunner:
    """Stateful, per-connection audio -> chord pipeline.

    Parameters
    ----------
    with_classifier:
        If ``False``, the runner emits CQT / onset messages but skips
        the CNN. Useful for development when you want to inspect the
        spectrogram without paying for inference.
    """

    def __init__(self, with_classifier: bool = True) -> None:
        self.buffer = AudioRingBuffer()
        self.cqt = CQTStream()
        self.onsets = OnsetDetector()
        self.segments = SegmentBuffer()
        self.classifier: ChordClassifier | None = None
        if with_classifier:
            self.classifier = ChordClassifier()

        # Wall-clock of the first ever audio sample. Used to map
        # segment column indices to "time since first sample" for the
        # browser.
        self._start_time: float | None = None
        # Last time we emitted a CQT update. Used for rate-limiting.
        self._last_cqt_emit_ms: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ingest_pcm(self, samples: np.ndarray) -> List[dict]:
        """Feed a Float32 mono chunk and collect outbound messages."""
        messages: List[dict] = []

        if self._start_time is None:
            self._start_time = time.monotonic()

        # 1. Append to the audio ring buffer.
        self.buffer.append(samples)
        total_appended = self.buffer.total_appended
        audio_tail = self.buffer.tail(config.AUDIO_MIN_SECONDS)

        # We need at least 2 s of audio before CQT / onsets are
        # meaningful.
        if self.buffer.available_seconds < config.AUDIO_MIN_SECONDS:
            return messages

        # 2. Update the CQT stream.
        new_cqt_cols, _start_sample = self.cqt.update(audio_tail, total_appended)
        if new_cqt_cols.size == 0:
            return messages

        # 3. Onset detection on the new slice.
        new_onset_frames, _envelope_tail = self.onsets.update(new_cqt_cols)

        # 4. Push to the segment buffer - this may produce one or more
        # completed windows.
        completed_windows = self.segments.push(new_cqt_cols, new_onset_frames)

        # 5. Run classification on each completed window.
        for window in completed_windows:
            if self.classifier is None:
                continue
            result = self.classifier.classify(window.cqt)
            onset_time = self._column_to_time(window.onset_frame)
            duration = window.duration_seconds
            messages.append(
                {
                    "type": "chord",
                    "raw_label": result["raw_label"],
                    "display_label": result["display_label"],
                    "confidence": result["confidence"],
                    "predicted_index": result["predicted_index"],
                    "onset_time": onset_time,
                    "duration": duration,
                    "truncated": window.truncated,
                    "source_frames": window.source_frames,
                }
            )

        # 6. CQT column update, rate-limited.
        now_ms = time.monotonic() * 1000.0
        if now_ms - self._last_cqt_emit_ms >= config.SEND_CQT_EVERY_MS:
            self._last_cqt_emit_ms = now_ms
            trail = self.cqt.columns[:, -config.CQT_TRAIL_COLUMNS :]
            messages.append(
                {
                    "type": "cqt_columns",
                    "n_bins": int(trail.shape[0]),
                    "n_cols": int(trail.shape[1]),
                    "time_s": self.buffer.available_seconds,
                    # Flatten in C order. The browser re-shapes to
                    # ``(n_bins, n_cols)``.
                    "columns": trail.astype(np.float32).flatten(order="C").tolist(),
                }
            )

        return messages

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _column_to_time(self, column_index: int) -> float:
        """Map a CQT column index to wall-clock seconds since start."""
        return column_index * config.CQT_HOP_LENGTH / config.AUDIO_SAMPLE_RATE

    def close(self) -> None:
        """Free the classifier (drops the Keras graph) and reset state."""
        self.classifier = None
        self.segments.reset()
        self.onsets.reset()
        self.buffer = AudioRingBuffer()
        self.cqt = CQTStream()


__all__ = ["PipelineRunner"]
