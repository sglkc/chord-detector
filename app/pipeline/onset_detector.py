"""Onset detection on the streaming CQT.

We compute a Superflux onset envelope (Boeck & Widmer 2013) on the
dB-scaled CQT and peak-pick with the same parameters as the offline
training pipeline. The detector keeps a "high water mark" of the last
emitted frame so each call returns only onsets that are *new*.

Superflux is always run on a trailing *context* window of stored CQT
columns (not on tiny per-tick slices alone), so lag / max_size have
enough history. Only the envelope samples that correspond to truly
new columns are appended. Both the CQT context and the envelope are
capped so memory stays bounded; ``last_emitted_frame`` is adjusted
when the envelope is trimmed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import librosa
import numpy as np

from .. import config

# Trailing CQT frames kept for Superflux context. Must cover lag /
# max_size plus a generous pad so edge effects of re-analysis stay
# far from the "new" tail we actually keep.
_CQT_CONTEXT_MAX: int = 512

# Envelope history cap: enough for peak-pick pre_avg / pre_max / wait
# windows plus headroom. Onset frame indices are mapped back to the
# global column coordinate via ``_envelope_start_frame``.
_ENVELOPE_MAX: int = 512


class OnsetDetector:
    """Streaming Superflux onset detector.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz.
    hop_length:
        Hop length used by the CQT (samples).
    superflux_params:
        Dict of kwargs forwarded to ``librosa.onset.onset_strength``.
    peak_pick_params:
        Dict of kwargs forwarded to ``librosa.onset.onset_detect``.

    Attributes
    ----------
    envelope: np.ndarray
        The running onset-strength envelope (capped). New envelope
        values are appended on every ``update`` call.
    last_emitted_frame: int
        Global column index of the last frame already returned to the
        caller. Used to compute the "new onsets only" diff.
    """

    def __init__(
        self,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        hop_length: int = config.CQT_HOP_LENGTH,
        superflux_params: dict = None,
        peak_pick_params: dict = None,
    ) -> None:
        self.sample_rate: int = int(sample_rate)
        self.hop_length: int = int(hop_length)
        self.superflux_params: dict = dict(superflux_params or config.SUPERFLUX_PARAMETERS)
        self.peak_pick_params: dict = dict(peak_pick_params or config.PEAK_PICK_PARAMETERS)

        self.envelope: np.ndarray = np.zeros(0, dtype=np.float32)
        self.last_emitted_frame: int = 0
        # Global column index corresponding to ``envelope[0]``.
        self._envelope_start_frame: int = 0
        # Global column index of the next CQT column we expect
        # (equals total columns ever received).
        self._next_global_column: int = 0
        # Trailing CQT history for Superflux context.
        self._cqt_history: Optional[np.ndarray] = None  # (n_bins, T) or None
        # Back-reference to the SegmentBuffer so ``set_param`` can
        # propagate debounce changes. Wired up by PipelineRunner.
        self.segments = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, cqt_db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Recompute the envelope and return newly detected onsets.

        Parameters
        ----------
        cqt_db:
            The latest CQT columns from ``CQTStream.update``, shape
            ``(n_bins, n_new_cols)``. These are appended to an internal
            trailing context; Superflux is run on that context so
            lag / max_size always see enough history.

        Returns
        -------
        new_onset_frames:
            1-D array of frame indices (in the *global* CQT column
            coordinate system) for onsets that arrived since the last
            call.
        new_envelope_tail:
            The freshly-computed envelope values for the new columns
            only (length ``n_new_cols``). Useful for visualization.
        """
        if cqt_db.size == 0 or cqt_db.ndim != 2 or cqt_db.shape[1] == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        n_new = int(cqt_db.shape[1])
        new_cols = cqt_db.astype(np.float32, copy=False)

        # ---- maintain trailing CQT context --------------------------------
        if self._cqt_history is None or self._cqt_history.size == 0:
            self._cqt_history = new_cols.copy()
        else:
            self._cqt_history = np.concatenate([self._cqt_history, new_cols], axis=1)
        if self._cqt_history.shape[1] > _CQT_CONTEXT_MAX:
            self._cqt_history = self._cqt_history[:, -_CQT_CONTEXT_MAX:].copy()

        # Superflux needs at least 2 frames to produce a meaningful flux.
        if self._cqt_history.shape[1] < 2:
            # Still advance the global column counter so indices stay
            # aligned once we have enough context.
            zeros = np.zeros(n_new, dtype=np.float32)
            self.envelope = np.concatenate([self.envelope, zeros])
            self._next_global_column += n_new
            self._trim_envelope()
            return np.zeros(0, dtype=np.int64), zeros

        # Onset strength on the trailing context. librosa expects a
        # power / magnitude spectrogram; we pass dB values and let it
        # convert internally. Superflux kwargs (lag, max_size) apply.
        context_envelope = librosa.onset.onset_strength(
            S=self._cqt_history,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            **self.superflux_params,
        ).astype(np.float32, copy=False)

        # Keep only the envelope samples that correspond to the truly
        # new columns (the tail of the context re-analysis).
        take = min(n_new, int(context_envelope.shape[0]))
        new_envelope = context_envelope[-take:]
        if take < n_new:
            # Extremely short context edge case: pad the leading new
            # frames with zeros so lengths stay consistent.
            pad = np.zeros(n_new - take, dtype=np.float32)
            new_envelope = np.concatenate([pad, new_envelope])

        self.envelope = np.concatenate([self.envelope, new_envelope])
        self._next_global_column += n_new
        self._trim_envelope()

        if self.envelope.size < 2:
            return np.zeros(0, dtype=np.int64), new_envelope

        # ``onset_detect`` on the running (capped) envelope so ``wait``
        # cooldowns stay correct. Map local indices back to global
        # column coordinates via ``_envelope_start_frame``.
        onset_frames_local = librosa.onset.onset_detect(
            onset_envelope=self.envelope,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=False,
            **self.peak_pick_params,
        )
        onset_frames = (
            np.asarray(onset_frames_local, dtype=np.int64) + int(self._envelope_start_frame)
        )

        # The first ``ONSET_WARMUP_FRAMES`` of *global* history are the
        # warmup window. Superflux often spuriously fires on the
        # transition from the zero-padded region at the very start, so
        # we drop any onsets that fall in the warmup zone.
        warmup_threshold = config.ONSET_WARMUP_FRAMES - 1
        new_onsets = onset_frames[
            (onset_frames > self.last_emitted_frame)
            & (onset_frames > warmup_threshold)
        ]
        if new_onsets.size:
            self.last_emitted_frame = int(new_onsets.max())

        return new_onsets.astype(np.int64, copy=False), new_envelope

    def envelope_at(self, global_column: int) -> float | None:
        """Superflux strength at a global CQT column, if still in history."""
        i = int(global_column) - int(self._envelope_start_frame)
        if i < 0 or i >= int(self.envelope.size):
            return None
        return float(self.envelope[i])

    def reset(self) -> None:
        """Drop the envelope, CQT context, and watermarks. Used on disconnect."""
        self.envelope = np.zeros(0, dtype=np.float32)
        self.last_emitted_frame = 0
        self._envelope_start_frame = 0
        self._next_global_column = 0
        self._cqt_history = None

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _trim_envelope(self) -> None:
        """Cap envelope length; keep global frame indices consistent."""
        if self.envelope.size <= _ENVELOPE_MAX:
            return
        trim = int(self.envelope.size - _ENVELOPE_MAX)
        self.envelope = self.envelope[trim:].copy()
        self._envelope_start_frame += trim
        # last_emitted_frame is in global coordinates and must not go
        # below the new start (onsets before the window are gone).
        if self.last_emitted_frame < self._envelope_start_frame:
            # Keep a watermark just before the retained window so we
            # do not re-emit peaks that may reappear at the left edge.
            self.last_emitted_frame = self._envelope_start_frame - 1

    # ------------------------------------------------------------------ #
    # Live tuning
    # ------------------------------------------------------------------ #

    # Keys that callers are allowed to mutate at runtime via the WS
    # ``set <key>=<value>`` control command. Whitelist keeps the
    # attack surface small.
    _SETTABLE_KEYS: frozenset = frozenset(
        {
            "min_onset_gap_ms",
            "peak_pick_delta",
            "peak_pick_wait",
        }
    )

    def set_param(self, key: str, value: str):
        """Update one runtime-tunable parameter, validating the key.

        Parameters
        ----------
        key:
            One of ``min_onset_gap_ms``, ``peak_pick_delta``,
            ``peak_pick_wait``. Snake-case on the wire so it
            matches the underscore-joined form used in
            ``app.config``.
        value:
            String form of the new value. Coerced to ``int`` or
            ``float`` based on the key. ``peak_pick_delta`` is
            prominence on the normalized [0, 1] envelope; ``"none"``
            resets it to 0.07.

        Returns
        -------
        The new value (typed). The caller echoes it back to the
        client as ``{"type": "param_updated", ...}``.

        Raises
        ------
        KeyError
            If ``key`` is not in the whitelist.
        ValueError
            If ``value`` cannot be coerced to the expected type.
        """
        if key not in self._SETTABLE_KEYS:
            raise KeyError(f"unknown parameter {key!r}")

        if key == "min_onset_gap_ms":
            ms = int(value)
            if ms < 0:
                raise ValueError("min_onset_gap_ms must be >= 0")
            # Recompute the debounce frame count and push it to the
            # linked SegmentBuffer so the next push uses the new gap.
            frames = max(
                1,
                int(
                    round(
                        ms * 0.001 * config.AUDIO_SAMPLE_RATE / config.CQT_HOP_LENGTH
                    )
                ),
            )
            self.min_onset_gap_frames = frames
            if self.segments is not None:
                self.segments.set_min_onset_gap_frames(frames)
            return ms

        if key == "peak_pick_delta":
            if value.strip().lower() in {"none", "null", ""}:
                self.peak_pick_params["delta"] = 0.07
                return 0.07
            f = float(value)
            if f < 0.0 or f > 1.0:
                raise ValueError(
                    "peak_pick_delta must be in [0, 1]; "
                    "the onset envelope is normalized"
                )
            self.peak_pick_params["delta"] = f
            return f

        if key == "peak_pick_wait":
            n = int(value)
            if n < 0:
                raise ValueError("peak_pick_wait must be >= 0")
            self.peak_pick_params["wait"] = n
            return n

        # Defensive - should be unreachable thanks to the whitelist.
        raise KeyError(key)  # pragma: no cover


__all__ = ["OnsetDetector"]
