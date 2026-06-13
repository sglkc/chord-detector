"""Streaming Constant-Q Transform.

We keep a small window of "freshly-analyzed" CQT columns and append to
it on every ``update()`` call. The implementation favors simplicity
over micro-optimization: we recompute the CQT on the last
``analysis_window_seconds`` of audio each tick and diff against the
previous result. On a typical CPU this is well below 30 ms for a 2 s
window @ 48 kHz, 216 bins, 36 bins/octave.

The output is the dB-scaled magnitude, exactly as the CNN expects
(``librosa.amplitude_to_db(..., ref=np.max)``).
"""

from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np

from .. import config


class CQTStream:
    """Maintains a rolling dB-scaled CQT of the live audio stream.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz.
    fmin:
        Lowest CQT bin (Hz). Must match training.
    n_bins:
        Number of CQT frequency bins (octaves * bins_per_octave).
    bins_per_octave:
        Frequency resolution of the CQT.
    hop_length:
        STFT/CQT hop length (samples).
    analysis_window_seconds:
        How much audio is fed into ``librosa.cqt`` per ``update`` call.
        Larger = more history kept, more CPU per tick. 2 s matches the
        minimum training window.

    Attributes
    ----------
    columns: np.ndarray
        Shape ``(n_bins, n_columns)``. Append-only within
        ``MAX_CQT_COLUMNS``; oldest columns are dropped on overflow.
    last_analyzed_sample: int
        Monotonic counter of the highest sample index already
        incorporated into ``columns``. Used to compute the diff on
        the next ``update`` call.
    """

    def __init__(
        self,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        fmin: float = config.CQT_FMIN,
        n_bins: int = config.CQT_FEATURE_BINS,
        bins_per_octave: int = config.CQT_BINS_PER_OCTAVE,
        hop_length: int = config.CQT_HOP_LENGTH,
        analysis_window_seconds: float = 2.0,
        max_columns: int = 4096,
    ) -> None:
        self.sample_rate: int = int(sample_rate)
        self.fmin: float = float(fmin)
        self.n_bins: int = int(n_bins)
        self.bins_per_octave: int = int(bins_per_octave)
        self.hop_length: int = int(hop_length)
        self.analysis_window_samples: int = int(round(analysis_window_seconds * self.sample_rate))
        self.max_columns: int = int(max_columns)

        # Empty CQT history. ``columns`` grows by ~``hop_length`` samples
        # worth of columns on every update (one column per hop).
        self.columns: np.ndarray = np.zeros((self.n_bins, 0), dtype=np.float32)
        self.last_analyzed_sample: int = 0  # monotonic

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(self, audio: np.ndarray, total_appended: int) -> Tuple[np.ndarray, int]:
        """Refresh the CQT using the latest slice of audio.

        Parameters
        ----------
        audio:
            1-D float32 mono audio (the full rolling buffer is fine -
            we only look at the tail).
        total_appended:
            The ``total_appended`` counter from the ``AudioRingBuffer``
            that produced ``audio``. Used as a watermark: we only
            re-analyze audio newer than ``last_analyzed_sample``.

        Returns
        -------
        new_columns:
            ``(n_bins, k)`` array of freshly computed CQT columns.
        start_sample:
            The sample index (in the original stream) corresponding to
            the *first* column of ``new_columns``. Used by callers to
            translate column indices back to time.
        """
        if audio.size < self.hop_length:
            return np.zeros((self.n_bins, 0), dtype=np.float32), self.last_analyzed_sample

        # We need at least ``analysis_window_samples`` of audio to
        # produce a stable CQT. We re-run CQT on the tail every tick -
        # this is the same pattern librosa's streaming recipes use and
        # keeps the per-tick cost bounded.
        window = audio[-self.analysis_window_samples :] if audio.size > self.analysis_window_samples else audio
        if window.size < self.hop_length * 4:
            return np.zeros((self.n_bins, 0), dtype=np.float32), self.last_analyzed_sample

        cqt_complex = librosa.cqt(
            y=window.astype(np.float32, copy=False),
            sr=self.sample_rate,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            hop_length=self.hop_length,
        )
        cqt_mag = np.abs(cqt_complex)
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max).astype(np.float32, copy=False)

        n_new_cols = cqt_db.shape[1]

        # Append. Truncate oldest columns to keep memory bounded.
        if self.columns.shape[1] + n_new_cols > self.max_columns:
            keep = self.max_columns - n_new_cols
            keep = max(keep, 0)
            self.columns = np.concatenate([self.columns[:, -keep:], cqt_db], axis=1) if keep else cqt_db.copy()
        else:
            self.columns = np.concatenate([self.columns, cqt_db], axis=1)

        # ``last_analyzed_sample`` is the absolute sample index of the
        # *last* column in this update. We don't track the precise
        # column-to-sample mapping (the audio buffer does that); the
        # offset between the rolling buffer's tail and the new columns
        # is ``window.size - n_new_cols * hop_length`` samples.
        start_sample = total_appended - window.size
        self.last_analyzed_sample = total_appended
        return cqt_db, start_sample

    # ------------------------------------------------------------------ #
    # Time helpers
    # ------------------------------------------------------------------ #

    def column_to_time(self, column_index: int) -> float:
        """Convert a column index into the CQT to absolute time (seconds).

        ``column_index`` is in the global column coordinate system (i.e.
        as returned by ``onset_detector``). 0 corresponds to the
        first column ever produced by this stream.
        """
        return column_index * self.hop_length / self.sample_rate

    @property
    def latest_column_index(self) -> int:
        """Index of the right-most (newest) column currently held."""
        return self.columns.shape[1] - 1


__all__ = ["CQTStream"]
