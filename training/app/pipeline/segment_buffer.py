"""Build classification windows from the live CQT stream.

State machine
-------------
``WAITING_FOR_ONSET`` (initial)
    Ignore incoming CQT columns. Wait for the first new onset.

``COLLECTING`` (entered on each new onset)
    Append each new CQT column to the current segment.

    * If the segment reaches ``CQT_FEATURE_FRAMES`` (188) columns,
      emit it as a *full* (non-truncated) window and return to
      ``WAITING_FOR_ONSET`` so the next onset starts a fresh chord.
    * If a *new* onset arrives while still collecting, the current
      segment is emitted *truncated* (whatever frames it has) and a
      fresh ``COLLECTING`` state begins at the new onset.

The actual classification (stretch + CNN) is done by
:class:`ChordClassifier` - this module only builds the windows and
emits them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from .. import config


class _State(str, Enum):
    WAITING_FOR_ONSET = "waiting_for_onset"
    COLLECTING = "collecting"


@dataclass
class SegmentWindow:
    """A CQT window ready to be fed to the classifier.

    Attributes
    ----------
    cqt: np.ndarray
        ``(CQT_FEATURE_BINS, source_frames)`` array. The classifier
        will linear-interpolate this to ``CQT_FEATURE_FRAMES``.
    onset_frame: int
        Column index in the CQT stream where the segment started.
    end_frame: int
        Column index where the segment stopped. Either
        ``onset_frame + CQT_FEATURE_FRAMES`` (full) or the index of
        the *next* onset (truncated).
    source_frames: int
        Number of frames actually collected. Always
        ``<= CQT_FEATURE_FRAMES``.
    truncated: bool
        ``True`` if the segment ended because another onset arrived
        before reaching 188 frames; ``False`` if it hit the natural
        188-frame boundary.
    """

    cqt: np.ndarray
    onset_frame: int
    end_frame: int
    source_frames: int
    truncated: bool

    @property
    def duration_seconds(self) -> float:
        """Audio duration represented by the segment, in seconds."""
        return self.source_frames * config.CQT_HOP_LENGTH / config.AUDIO_SAMPLE_RATE


class SegmentBuffer:
    """Builds ``SegmentWindow`` objects from a stream of CQT columns.

    The buffer is column-oriented. The caller pushes columns one at a
    time (or as small arrays) along with any newly-detected onsets.
    """

    def __init__(
        self,
        target_frames: int = config.CQT_FEATURE_FRAMES,
        min_onset_gap_ms: int = config.MIN_ONSET_GAP_MS,
    ) -> None:
        self.target_frames: int = int(target_frames)
        # Onset-frame debounce. Translates to a frame count using the
        # standard hop length / sample rate.
        self.min_onset_gap_frames: int = max(
            1, int(round(min_onset_gap_ms * 0.001 * config.AUDIO_SAMPLE_RATE / config.CQT_HOP_LENGTH))
        )

        self._state: _State = _State.WAITING_FOR_ONSET
        self._current_columns: List[np.ndarray] = []
        self._current_onset_frame: int = 0
        self._last_emitted_onset_frame: int = -10_000  # effectively no debounce at start

    def set_min_onset_gap_frames(self, frame_count: int) -> None:
        """Live-update the debounce. Called by ``OnsetDetector.set_param``."""
        self.min_onset_gap_frames = max(1, int(frame_count))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._state = _State.WAITING_FOR_ONSET
        self._current_columns = []
        self._last_emitted_onset_frame = -10_000

    def push(
        self,
        new_columns: np.ndarray,
        new_onset_frames: np.ndarray,
    ) -> List[SegmentWindow]:
        """Push new CQT columns and onsets; return any completed segments.

        Parameters
        ----------
        new_columns:
            Shape ``(n_bins, k)`` - the trailing CQT columns.
        new_onset_frames:
            Frame indices of onsets detected in the *global* column
            coordinate system.

        Returns
        -------
        list[SegmentWindow]
            Zero or more completed windows ready for classification.
        """
        completed: List[SegmentWindow] = []

        # If we have new onsets and we're collecting, the current
        # segment is truncated. The first new onset also starts a new
        # collecting window. The "current segment's end" is the column
        # index of the *earliest* new onset frame, and we keep columns
        # up to (but not including) that onset.
        for onset_frame in new_onset_frames:
            # Debounce: skip onsets that are too close to the previous one.
            if (onset_frame - self._last_emitted_onset_frame) < self.min_onset_gap_frames:
                continue

            if self._state is _State.COLLECTING:
                truncated = self._finalize_current(onset_frame)
                if truncated is not None:
                    completed.append(truncated)
            else:
                # WAITING_FOR_ONSET -> COLLECTING
                self._state = _State.COLLECTING
                self._current_columns = []
                self._current_onset_frame = int(onset_frame)

            self._last_emitted_onset_frame = int(onset_frame)

        # Append new CQT columns to the collecting buffer.
        if self._state is _State.COLLECTING and new_columns.size:
            # We only care about columns from ``_current_onset_frame``
            # onward, but for simplicity we just append them all and
            # track the total length.
            self._current_columns.append(new_columns.astype(np.float32, copy=False))

            total_frames = sum(c.shape[1] for c in self._current_columns)
            if total_frames >= self.target_frames:
                full = self._finalize_current(self._current_onset_frame + self.target_frames)
                if full is not None:
                    completed.append(full)
                # Once we've emitted a full window we go back to
                # waiting. The next onset will start a new chord. This
                # matches the offline training pipeline.
                self._state = _State.WAITING_FOR_ONSET
                self._current_columns = []

        return completed

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _finalize_current(self, end_frame: int) -> Optional[SegmentWindow]:
        """Concatenate the current columns, crop, and emit a window."""
        if not self._current_columns:
            return None

        full = np.concatenate(self._current_columns, axis=1)  # (n_bins, T)
        # Crop to the target length. ``end_frame - onset_frame`` is
        # the number of frames this segment should contain.
        target_length = min(end_frame - self._current_onset_frame, self.target_frames)
        target_length = max(0, target_length)
        cropped = full[:, :target_length]
        if cropped.shape[1] == 0:
            return None

        truncated = cropped.shape[1] < self.target_frames
        return SegmentWindow(
            cqt=cropped,
            onset_frame=self._current_onset_frame,
            end_frame=self._current_onset_frame + cropped.shape[1],
            source_frames=int(cropped.shape[1]),
            truncated=truncated,
        )


__all__ = ["SegmentBuffer", "SegmentWindow"]
