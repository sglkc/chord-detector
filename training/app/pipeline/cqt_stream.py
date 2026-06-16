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

        # Pre-build the constant-Q kernel once so the first
        # ``librosa.cqt`` call doesn't pay the ~1.5 s kernel
        # construction cost. In librosa >= 0.10 the ``cqt(filter=...)``
        # fast path is gone, but the kernel is still cached
        # internally on subsequent calls. We trigger that cache by
        # running cqt on a tiny dummy signal and discarding the
        # result - cheaper than waiting for the first real audio
        # chunk to amortize the build.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = librosa.cqt(
                y=np.zeros(int(0.05 * self.sample_rate), dtype=np.float32),
                sr=self.sample_rate,
                fmin=self.fmin,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
                hop_length=self.hop_length,
                sparsity=0.01,
            )

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
            ``(n_bins, k)`` array of freshly computed CQT columns -
            ONLY the columns whose sample indices lie after
            ``last_analyzed_sample`` are returned and appended to
            ``self.columns``. This keeps the column count consistent
            with the number of samples that have actually arrived.
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
            # ``sparsity`` enables the sparse matrix-multiply path
            # in librosa which is the documented fast path for
            # full-band CQTs (>= 84 bins) on long windows. With a
            # 216-bin / 2 s window it is ~2x faster than the dense
            # default. See librosa PR #1271.
            sparsity=0.01,
        )
        cqt_mag = np.abs(cqt_complex)
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max).astype(np.float32, copy=False)

        # ``start_sample`` of the *whole* recomputed CQT = the sample
        # index of the first sample in ``window``. The CQT's column k
        # covers sample ``start_sample + k * hop_length``.
        window_start = total_appended - window.size

        # Drop the columns that are already represented in
        # ``self.columns``: anything at or before ``last_analyzed_sample``
        # is a duplicate from the previous call.
        if self.last_analyzed_sample <= window_start:
            # No overlap - return everything.
            new_cols = cqt_db
        else:
            # The first "new" CQT column is the one whose sample index
            # is just past ``last_analyzed_sample``. That column's index
            # in ``cqt_db`` is::
            #
            #   k = ceil((last_analyzed_sample - window_start + 1) / hop_length)
            #
            # which we clamp to ``[0, n_cols]``.
            import math

            k = int(math.ceil((self.last_analyzed_sample - window_start + 1) / self.hop_length))
            k = max(0, min(k, cqt_db.shape[1]))
            new_cols = cqt_db[:, k:]

        if new_cols.shape[1] == 0:
            # Nothing genuinely new (e.g. only a single chunk arrived
            # in a tick that produced no fresh columns because of the
            # CQT's internal overlap).
            return np.zeros((self.n_bins, 0), dtype=np.float32), self.last_analyzed_sample

        start_sample = window_start + (cqt_db.shape[1] - new_cols.shape[1]) * self.hop_length

        # Append. Truncate oldest columns to keep memory bounded.
        if self.columns.shape[1] + new_cols.shape[1] > self.max_columns:
            keep = self.max_columns - new_cols.shape[1]
            keep = max(keep, 0)
            self.columns = (
                np.concatenate([self.columns[:, -keep:], new_cols], axis=1)
                if keep
                else new_cols.copy()
            )
        else:
            self.columns = np.concatenate([self.columns, new_cols], axis=1)

        # Track the absolute sample index of the *last* CQT column we
        # just produced. This is the watermark for the next call.
        self.last_analyzed_sample = start_sample + (new_cols.shape[1] - 1) * self.hop_length
        return new_cols, start_sample

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
