"""Top-level pipeline orchestrator.

One :class:`PipelineRunner` is created per WebSocket connection. It
owns the audio buffer, CQT stream, onset detector and segment buffer
(per-connection DSP state). The CNN classifier may be injected as a
shared instance loaded once at app startup, or loaded privately (e.g.
in unit tests).

The runner returns a list of *messages* to ship back to the browser.
The message types are:

``{"type": "cqt_columns", "columns": [...], "n_cols": int, "time_s": float}``

    A 1-D flat array of ``n_bins * n_cols`` floats representing the
    trailing CQT columns (dB). The browser paints them right-to-left
    on a scrolling canvas.

``{"type": "onset", "column": int, "time_s": float, "strength": float}``

    A newly detected Superflux onset. ``column`` is the global CQT
    column index (same coordinate system as ``end_column``). The
    browser draws it as a vertical marker on the spectrogram.
    ``strength`` is the Superflux envelope value at that peak.

``{"type": "chord", "raw_label": str, "display_label": str,
   "confidence": float, "onset_time": float, "duration": float,
   "truncated": bool, "source_frames": int, "onset_column": int,
   "strength": float}``

    The result of a CNN inference on a completed segment.
    ``onset_column`` is the global CQT column of the segment start
    (same coordinate system as ``onset.column`` / ``end_column``)
    so the browser can pin the label to that spectrogram marker.

The runner is intentionally synchronous (no asyncio) - the WebSocket
handler is expected to call it inside ``asyncio.to_thread``. Callers
must not overlap ``ingest_pcm`` with ``reset`` / ``set_param`` on the
same instance (the server serializes these onto one lane).
"""

from __future__ import annotations

import time
from typing import List, Optional, Union

import numpy as np

from .. import config
from .audio_buffer import AudioRingBuffer
from .classifier import ChordClassifier
from .cqt_stream import CQTStream
from .onset_detector import OnsetDetector
from .segment_buffer import SegmentBuffer

# Sentinel: distinguish "caller did not pass classifier" from
# "caller explicitly injected None" (e.g. shared load failed).
_CLASSIFIER_UNSET = object()


class PipelineRunner:
    """Stateful, per-connection audio -> chord pipeline.

    Parameters
    ----------
    with_classifier:
        If ``False``, the runner emits CQT / onset messages but skips
        the CNN. Useful for development when you want to inspect the
        spectrogram without paying for inference. Ignored when the
        caller passes ``classifier=...`` (including ``classifier=None``).
    classifier:
        Pre-loaded :class:`ChordClassifier`, or ``None`` to mean
        "shared injection with no model" (do **not** private-load).
        Omit the argument entirely to private-load when
        ``with_classifier=True`` (unit tests).
    model_status:
        Optional status dict to surface to the client (``loaded``,
        ``load_time_s``, ``error``). When the caller injects a
        classifier slot (even if ``None``), this status is preferred
        over a private load attempt.
    """

    def __init__(
        self,
        with_classifier: bool = True,
        classifier: Union[ChordClassifier, None, object] = _CLASSIFIER_UNSET,
        model_status: Optional[dict] = None,
    ) -> None:
        self.buffer = AudioRingBuffer()
        self.cqt = CQTStream()
        self.onsets = OnsetDetector()
        self.segments = SegmentBuffer()
        self.classifier: ChordClassifier | None = None
        # Model-load status. Surfaced to the client so it can show a
        # "model: loading..." / ready / error pill.
        self.model_status: dict = {
            "loaded": False,
            "load_time_s": 0.0,
            "error": None,
        }

        if classifier is not _CLASSIFIER_UNSET:
            # Explicit injection path (shared app instance, possibly
            # None if startup load failed). Never private-load here.
            self.classifier = classifier  # type: ignore[assignment]
            if model_status is not None:
                self.model_status = dict(model_status)
            else:
                self.model_status = {
                    "loaded": self.classifier is not None,
                    "load_time_s": 0.0,
                    "error": None
                    if self.classifier is not None
                    else "classifier injected as None",
                }
        elif with_classifier:
            # No injection: private load for standalone / tests.
            self._load_classifier()

        # Keep SegmentBuffer's debounce in sync with the detector's
        # live-tuned value.
        self.onsets.segments = self.segments

        # Wall-clock of the first ever audio sample. Used to map
        # segment column indices to "time since first sample" for the
        # browser.
        self._start_time: float | None = None
        # Last time we emitted a CQT update. Used for rate-limiting.
        self._last_cqt_emit_ms: float = 0.0

    def _load_classifier(self) -> None:
        """Load the Keras CNN, recording timing / error on self.model_status."""
        import time as _time
        t0 = _time.monotonic()
        try:
            self.classifier = ChordClassifier()
            self.model_status = {
                "loaded": True,
                "load_time_s": _time.monotonic() - t0,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - any TF / IO error is fatal
            self.classifier = None
            self.model_status = {
                "loaded": False,
                "load_time_s": _time.monotonic() - t0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def reset(self) -> None:
        """Drop in-flight state so the next chunk starts fresh."""
        self.segments.reset()
        self.onsets.reset()
        self.buffer = AudioRingBuffer()
        self.cqt = CQTStream()
        self._start_time = None
        self._last_cqt_emit_ms = 0.0

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

        # 3. Onset detection on the new columns (detector keeps its own
        # trailing CQT context for Superflux).
        new_onset_frames, _envelope_tail = self.onsets.update(new_cqt_cols)

        # Emit each new onset immediately (not rate-limited with CQT)
        # so the browser can mark chord start/end on the spectrogram
        # even when this tick does not send a cqt_columns snapshot.
        for onset_frame in np.asarray(new_onset_frames, dtype=np.int64).ravel():
            onset_frame = int(onset_frame)
            messages.append(
                {
                    "type": "onset",
                    "column": onset_frame,
                    "time_s": self._column_to_time(onset_frame),
                    "strength": self.onsets.envelope_at(onset_frame),
                }
            )

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
                    "onset_column": int(window.onset_frame),
                    "strength": self.onsets.envelope_at(window.onset_frame),
                }
            )

        # 6. CQT column update, rate-limited. Always send a full trail
        # snapshot (last CQT_TRAIL_COLUMNS), not a delta.
        now_ms = time.monotonic() * 1000.0
        if now_ms - self._last_cqt_emit_ms >= config.SEND_CQT_EVERY_MS:
            self._last_cqt_emit_ms = now_ms
            trail = self.cqt.columns[:, -config.CQT_TRAIL_COLUMNS :]
            messages.append(
                {
                    "type": "cqt_columns",
                    "n_bins": int(trail.shape[0]),
                    "n_cols": int(trail.shape[1]),
                    # Monotonic end index of the CQT stream so the client
                    # can append only *new* columns and keep a full-width
                    # scrolling history on the canvas.
                    "end_column": int(self.cqt.total_columns),
                    "time_s": self.buffer.available_seconds,
                    # Flatten in C order: index = b * n_cols + c.
                    # The browser re-shapes to ``(n_bins, n_cols)``.
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
        """Release per-connection DSP state and drop the classifier ref.

        Only drops the local reference to the classifier. A shared
        instance held by the app lifespan stays alive; a private
        instance becomes eligible for GC when no other refs remain.
        """
        self.classifier = None
        self.segments.reset()
        self.onsets.reset()
        self.buffer = AudioRingBuffer()
        self.cqt = CQTStream()


__all__ = ["PipelineRunner"]
