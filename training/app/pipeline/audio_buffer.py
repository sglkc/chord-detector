"""Bounded ring buffer for the live mono audio stream.

The browser sends ~85 ms chunks of Float32 audio (48 kHz, mono). The
backend concatenates them, drops anything older than
``MAX_AUDIO_SECONDS`` and exposes a small read API used by the rest of
the pipeline.

The class is deliberately tiny: pure NumPy, no asyncio, no locks. The
WebSocket handler runs in a single asyncio task, so there is no
concurrent mutation to worry about.
"""

from __future__ import annotations

import numpy as np

from .. import config


class AudioRingBuffer:
    """Rolling buffer of mono Float32 audio.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz. Used to convert seconds <-> samples.
    max_seconds:
        Hard upper bound on buffered audio. Older samples are dropped
        on every ``append`` call.

    Attributes
    ----------
    sample_rate: int
    max_samples: int
    samples: np.ndarray
        Underlying 1-D float32 buffer. Length is always
        ``<= max_samples``.
    total_appended: int
        Monotonic counter of *all* samples ever received (never
        decremented). The CQT / onset stages use this as a watermark
        so they only consume the freshly-arrived slice.
    """

    def __init__(
        self,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        max_seconds: float = config.MAX_AUDIO_SECONDS,
    ) -> None:
        self.sample_rate: int = int(sample_rate)
        self.max_samples: int = int(round(max_seconds * self.sample_rate))
        # Pre-allocate the full ring once. Cheaper than repeated
        # ``np.concatenate`` and ``np.pad`` calls.
        self.samples: np.ndarray = np.zeros(self.max_samples, dtype=np.float32)
        # ``write_pos`` is the next free index in ``samples``. We treat
        # the buffer as a circular array but, on every ``append`` we
        # re-linearize it via ``np.roll`` only when the wrap point is
        # crossed. This keeps the contiguous "tail" cheap to read.
        self.write_pos: int = 0
        self.total_appended: int = 0  # total samples ever appended

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def append(self, chunk: np.ndarray) -> None:
        """Append a Float32 mono chunk to the buffer.

        Older samples (beyond ``max_samples``) are dropped. The
        underlying array is rolled left only when the write cursor
        wraps around.
        """
        if chunk.ndim != 1:
            # Defensive: should never happen with the WebSocket path
            # but a 2-D array would silently broadcast to a 0-D sum.
            raise ValueError(f"AudioRingBuffer expects 1-D mono audio, got shape {chunk.shape}")

        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32, copy=False)

        n = chunk.shape[0]
        if n == 0:
            return

        # If the new chunk is larger than the entire buffer, only the
        # tail survives.
        if n >= self.max_samples:
            self.samples[:] = chunk[-self.max_samples :]
            self.write_pos = 0
            self.total_appended += n
            return

        end = self.write_pos + n
        if end <= self.max_samples:
            # No wrap - simple slice write.
            self.samples[self.write_pos : end] = chunk
            self.write_pos = end if end < self.max_samples else 0
        else:
            # Wrap around the end. Split into two slices.
            first = self.max_samples - self.write_pos
            self.samples[self.write_pos :] = chunk[:first]
            self.samples[: n - first] = chunk[first:]
            self.write_pos = n - first

        self.total_appended += n

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def tail(self, seconds: float) -> np.ndarray:
        """Return the most recent ``seconds`` of audio.

        The returned array is a *copy*; mutating it is safe. The
        length may be less than ``seconds * sample_rate`` while the
        initial 2-second warm-up is still filling.
        """
        n_desired = int(round(seconds * self.sample_rate))
        n_desired = max(0, min(n_desired, self.max_samples))
        if n_desired == 0:
            return np.zeros(0, dtype=np.float32)

        # We rely on ``total_appended`` (monotonic) to decide if the
        # buffer is still warming up. If the producer is still
        # under-running, the contiguous head may be empty.
        n_available = min(self.total_appended, self.max_samples)
        if n_desired > n_available:
            n_desired = n_available

        if n_desired == 0:
            return np.zeros(0, dtype=np.float32)

        end = self.write_pos
        start = end - n_desired
        if start >= 0:
            return self.samples[start:end].copy()
        # Wrapped case.
        return np.concatenate([self.samples[start:], self.samples[:end]]).copy()

    @property
    def available_seconds(self) -> float:
        """How many seconds of audio are currently buffered."""
        return min(self.total_appended, self.max_samples) / self.sample_rate


__all__ = ["AudioRingBuffer"]
