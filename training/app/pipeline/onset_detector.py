"""Onset detection on the streaming CQT.

We compute a Superflux onset envelope (Boeck & Widmer 2013) on the
dB-scaled CQT and peak-pick with the same parameters as the offline
training pipeline. The detector keeps a "high water mark" of the last
emitted frame so each call returns only onsets that are *new*.
"""

from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np

from .. import config


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
        The full running onset-strength envelope. New envelope values
        are appended on every ``update`` call.
    last_emitted_frame: int
        Index of the last frame already returned to the caller. Used
        to compute the "new onsets only" diff.
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
            ``(n_bins, n_new_cols)``.

        Returns
        -------
        new_onset_frames:
            1-D array of frame indices (in the CQT column coordinate
            system) for onsets that arrived since the last call.
        new_envelope_tail:
            The freshly-computed envelope values (same length as
            ``cqt_db.shape[1]``). Useful for visualization.
        """
        if cqt_db.size == 0 or cqt_db.shape[1] < 2:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        # Onset strength on the CQT magnitude. librosa expects a
        # power / magnitude spectrogram; we pass dB values and let it
        # convert internally. The Superflux-specific kwargs (lag,
        # max_size) are applied automatically.
        new_envelope = librosa.onset.onset_strength(
            S=cqt_db,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            **self.superflux_params,
        ).astype(np.float32, copy=False)

        # ``onset_detect`` is run on the *running* envelope so it
        # maintains the ``wait`` cooldown correctly. We then filter to
        # only the onsets that arrived after ``last_emitted_frame``.
        self.envelope = np.concatenate([self.envelope, new_envelope])

        if self.envelope.size < 2:
            return np.zeros(0, dtype=np.int64), new_envelope

        onset_frames = librosa.onset.onset_detect(
            onset_envelope=self.envelope,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=False,
            **self.peak_pick_params,
        )

        # The first ``ONSET_WARMUP_FRAMES`` of envelope correspond to
        # the warmup window (no real audio is being listened to yet).
        # Superflux often spuriously fires on the transition from the
        # zero-padded region at the very start of the envelope, so we
        # drop any onsets that fall in the warmup zone.
        warmup_threshold = config.ONSET_WARMUP_FRAMES - 1
        new_onsets = onset_frames[
            (onset_frames > self.last_emitted_frame)
            & (onset_frames > warmup_threshold)
        ]
        if new_onsets.size:
            self.last_emitted_frame = int(new_onsets.max())

        return new_onsets.astype(np.int64, copy=False), new_envelope

    def reset(self) -> None:
        """Drop the envelope and watermark. Used on disconnect."""
        self.envelope = np.zeros(0, dtype=np.float32)
        self.last_emitted_frame = 0

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
            ``float`` based on the key. ``peak_pick_delta`` accepts
            a numeric value or the literal ``"None"`` (case
            insensitive) to clear the override and fall back to
            librosa's default.

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
                self.peak_pick_params.pop("delta", None)
                return None
            f = float(value)
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
