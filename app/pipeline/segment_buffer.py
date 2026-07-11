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

    The buffer is column-oriented. The caller pushes columns as small
    arrays along with any newly-detected onsets. A running global
    column index ensures mid-batch onsets only keep columns at/after
    the current onset frame (pre-onset columns from the same batch are
    discarded).

    Late onsets (``onset_frame`` already covered by previously
    collected columns) split the buffer: frames before the onset form
    the truncated window, and the remainder seeds the new segment.
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
        # Global index of the next column that will be pushed (0-based
        # stream coordinate, matching onset frame indices).
        self._next_global_column: int = 0

    def set_min_onset_gap_frames(self, frame_count: int) -> None:
        """Live-update the debounce. Called by ``OnsetDetector.set_param``."""
        self.min_onset_gap_frames = max(1, int(frame_count))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._state = _State.WAITING_FOR_ONSET
        self._current_columns = []
        self._current_onset_frame = 0
        self._last_emitted_onset_frame = -10_000
        self._next_global_column = 0

    def push(
        self,
        new_columns: np.ndarray,
        new_onset_frames: np.ndarray,
    ) -> List[SegmentWindow]:
        """Push new CQT columns and onsets; return any completed segments.

        Parameters
        ----------
        new_columns:
            Shape ``(n_bins, k)`` - the trailing CQT columns, in stream
            order. Column 0 of this array has global index
            ``_next_global_column``.
        new_onset_frames:
            Frame indices of onsets detected in the *global* column
            coordinate system.

        Returns
        -------
        list[SegmentWindow]
            Zero or more completed windows ready for classification.
        """
        completed: List[SegmentWindow] = []
        n_new = int(new_columns.shape[1]) if new_columns.size else 0
        batch_start = self._next_global_column
        batch_end = batch_start + n_new

        # Debounced, sorted onset list for this call.
        pending_onsets: List[int] = []
        for onset_frame in np.asarray(new_onset_frames, dtype=np.int64).ravel():
            onset_frame = int(onset_frame)
            last = (
                pending_onsets[-1]
                if pending_onsets
                else self._last_emitted_onset_frame
            )
            if (onset_frame - last) < self.min_onset_gap_frames:
                continue
            pending_onsets.append(onset_frame)

        # Absolute stream index of the next column not yet consumed
        # from this batch.
        cursor_abs = batch_start

        for onset_frame in pending_onsets:
            # Columns strictly before this onset belong to the current
            # segment (if any). The onset column starts the next one.
            if self._state is _State.COLLECTING and n_new:
                self._append_global_range(
                    new_columns,
                    batch_start,
                    lo=max(cursor_abs, self._current_onset_frame),
                    hi=min(onset_frame, batch_end),
                    completed=completed,
                )
            cursor_abs = max(cursor_abs, min(onset_frame, batch_end))
            self._handle_onset(onset_frame, completed)

        # Trailing columns after the last onset (or the whole batch if
        # there were no onsets this call).
        if self._state is _State.COLLECTING and n_new:
            self._append_global_range(
                new_columns,
                batch_start,
                lo=max(cursor_abs, self._current_onset_frame),
                hi=batch_end,
                completed=completed,
            )

        self._next_global_column = batch_end
        return completed

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _handle_onset(self, onset_frame: int, completed: List[SegmentWindow]) -> None:
        """Finalize (if needed) and start a new COLLECTING window.

        If the buffer already holds columns past ``onset_frame`` (late
        peak-pick reporting an earlier onset), split so the remainder
        seeds the new segment instead of being discarded.
        """
        if self._state is _State.COLLECTING:
            remainder = self._split_current_at(onset_frame)
            truncated = self._finalize_current(onset_frame)
            if truncated is not None:
                completed.append(truncated)
            # Begin collecting at the new onset; seed with any frames
            # already buffered at/after that onset.
            self._current_columns = remainder
            self._current_onset_frame = int(onset_frame)
            # Stay in COLLECTING.
        else:
            # WAITING_FOR_ONSET -> COLLECTING
            self._state = _State.COLLECTING
            self._current_columns = []
            self._current_onset_frame = int(onset_frame)
        self._last_emitted_onset_frame = int(onset_frame)

    def _split_current_at(self, onset_frame: int) -> List[np.ndarray]:
        """Split collected columns at ``onset_frame``.

        Assumes the concatenated buffer is contiguous in global time
        starting at ``_current_onset_frame`` (the append path only
        stores columns at/after the current onset, in order).

        After return, ``_current_columns`` holds only frames strictly
        before ``onset_frame`` (for finalize). The returned list holds
        frames at/after ``onset_frame`` (for the new segment).
        """
        if not self._current_columns:
            return []

        full = np.concatenate(self._current_columns, axis=1)
        # full[:, k] <-> global frame _current_onset_frame + k
        n_before = int(onset_frame) - int(self._current_onset_frame)
        if n_before <= 0:
            # Onset at or before the segment start: nothing for the old
            # window; entire buffer belongs to the new segment.
            self._current_columns = []
            return [full] if full.shape[1] > 0 else []

        if n_before >= full.shape[1]:
            # Onset is at/after the end of what we have collected; all
            # frames belong to the old window, no remainder.
            self._current_columns = [full]
            return []

        old = full[:, :n_before]
        rem = full[:, n_before:]
        self._current_columns = [old]
        return [rem]

    def _append_global_range(
        self,
        new_columns: np.ndarray,
        batch_start: int,
        lo: int,
        hi: int,
        completed: List[SegmentWindow],
    ) -> None:
        """Append columns whose global indices lie in ``[lo, hi)``."""
        if self._state is not _State.COLLECTING:
            return
        if hi <= lo:
            return
        c0 = lo - batch_start
        c1 = hi - batch_start
        if c1 <= c0:
            return
        self._current_columns.append(
            new_columns[:, c0:c1].astype(np.float32, copy=False)
        )
        self._maybe_emit_full(completed)

    def _maybe_emit_full(self, completed: List[SegmentWindow]) -> None:
        """Emit a full window if we have collected ``target_frames``."""
        if self._state is not _State.COLLECTING:
            return
        total_frames = sum(c.shape[1] for c in self._current_columns)
        if total_frames < self.target_frames:
            return
        full = self._finalize_current(self._current_onset_frame + self.target_frames)
        if full is not None:
            completed.append(full)
        # Full window -> wait for the next onset (matches offline).
        # Drop any overshoot past target_frames (not needed for CNN).
        self._state = _State.WAITING_FOR_ONSET
        self._current_columns = []

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
