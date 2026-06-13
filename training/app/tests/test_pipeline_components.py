"""Component-level tests for the streaming pipeline.

Run with:

    .venv/bin/python -m app.tests.test_pipeline_components

These tests exercise the pipeline modules in isolation so we can iterate
on the signal-processing logic without spinning up the full FastAPI
server. They are intentionally synchronous and CPU-only and use only
the standard library's ``unittest`` to avoid extra dependencies.
"""

from __future__ import annotations

import unittest

import numpy as np

from app.config import (
    AUDIO_SAMPLE_RATE,
    CQT_BINS_PER_OCTAVE,
    CQT_FEATURE_BINS,
    CQT_FEATURE_FRAMES,
    CQT_HOP_LENGTH,
    MODEL_LABELS,
)
from app.pipeline.audio_buffer import AudioRingBuffer
from app.pipeline.cqt_stream import CQTStream
from app.pipeline.onset_detector import OnsetDetector
from app.pipeline.segment_buffer import SegmentBuffer


# ---------------------------------------------------------------------------
# AudioRingBuffer
# ---------------------------------------------------------------------------


class AudioRingBufferTests(unittest.TestCase):
    def test_append_and_tail(self):
        buf = AudioRingBuffer(sample_rate=48000, max_seconds=1.0)
        self.assertEqual(buf.available_seconds, 0.0)

        chunk = np.ones(4800, dtype=np.float32) * 0.5  # 0.1 s
        buf.append(chunk)
        self.assertAlmostEqual(buf.available_seconds, 0.1, places=3)

        tail = buf.tail(0.05)
        self.assertEqual(tail.shape, (2400,))
        self.assertTrue(np.all(tail == 0.5))

    def test_wraps_around(self):
        # Three appends of 2000 samples into a 4800-cap ring:
        #   - i=0: 0..1999  -> samples[0:2000]
        #   - i=1: 2000..3999 -> samples[2000:4000]
        #   - i=2: 4000..5999 wraps, 800 of 2 at samples[4000:4800],
        #          1200 of 2 at samples[0:1200]
        # After the wrap, samples[0:800] still holds the original zeros
        # and samples[1200:2000] still holds 1.0 (both never overwritten).
        # ``tail(0.1)`` re-stitches by write_pos=1200 and returns the
        # most-recent 4800 samples in chronological order, so:
        #   800 zeros, 2000 ones, 2000 twos.
        buf = AudioRingBuffer(sample_rate=48000, max_seconds=0.1)
        for i in range(3):
            buf.append(np.full(2000, float(i), dtype=np.float32))
        self.assertAlmostEqual(buf.available_seconds, 0.1, places=3)
        tail = buf.tail(0.1)
        self.assertEqual(tail.shape, (4800,))
        self.assertTrue(np.all(tail[:800] == 0.0))
        self.assertTrue(np.all(tail[800:2800] == 1.0))
        self.assertTrue(np.all(tail[2800:] == 2.0))


# ---------------------------------------------------------------------------
# CQTStream
# ---------------------------------------------------------------------------


class CQTStreamTests(unittest.TestCase):
    def test_columns_have_correct_shape(self):
        import librosa

        sr = AUDIO_SAMPLE_RATE
        rng = np.random.default_rng(0)
        y = rng.standard_normal(3 * sr).astype(np.float32) * 0.05

        stream = CQTStream(
            sample_rate=sr,
            fmin=librosa.note_to_hz("C1"),
            n_bins=CQT_FEATURE_BINS,
            bins_per_octave=CQT_BINS_PER_OCTAVE,
            hop_length=CQT_HOP_LENGTH,
            analysis_window_seconds=2.0,
        )
        new_cols, _start = stream.update(y, total_appended=len(y))
        self.assertEqual(new_cols.shape[0], CQT_FEATURE_BINS)
        expected = int(2.0 * sr / CQT_HOP_LENGTH)
        self.assertLess(abs(new_cols.shape[1] - expected), 5)
        self.assertTrue(np.isfinite(new_cols).all())
        # CQT is in dB scale. After amplitude_to_db with ref=np.max, the
        # global maximum is exactly 0, so the array is mostly <= 0 with
        # possibly a few 0s. The maximum must not be > 0.
        self.assertLessEqual(float(new_cols.max()), 0.0 + 1e-6)


# ---------------------------------------------------------------------------
# OnsetDetector
# ---------------------------------------------------------------------------


class OnsetDetectorTests(unittest.TestCase):
    def test_finds_click_in_noise(self):
        import librosa

        sr = AUDIO_SAMPLE_RATE
        # 6 seconds of audio: 2s warmup + 4s after. The burst is placed
        # well past the warmup window so the warmup filter does not
        # discard it.
        rng = np.random.default_rng(1)
        noise = rng.standard_normal(6 * sr).astype(np.float32) * 0.01
        burst = np.zeros(2048, dtype=np.float32)
        burst[512:1536] = np.hanning(1024) * 0.8
        # CQT column = sample / 512; place burst around column 250
        # (audio time ~2.67s).
        burst_start_sample = 250 * 512
        noise[burst_start_sample : burst_start_sample + 2048] += burst

        det = OnsetDetector(
            sample_rate=sr,
            hop_length=CQT_HOP_LENGTH,
            superflux_params={"lag": 2, "max_size": 3},
            peak_pick_params={
                "pre_max": 30,
                "post_max": 1,
                "pre_avg": 100,
                "post_avg": 1,
                "wait": 30,
            },
        )
        cqt = librosa.amplitude_to_db(
            np.abs(
                librosa.cqt(
                    noise,
                    sr=sr,
                    fmin=librosa.note_to_hz("C1"),
                    n_bins=CQT_FEATURE_BINS,
                    bins_per_octave=CQT_BINS_PER_OCTAVE,
                    hop_length=CQT_HOP_LENGTH,
                )
            ),
            ref=np.max,
        )
        new_onsets, _env = det.update(cqt)
        # The detector must ignore onsets in the warmup window
        # (frames 0..CQT_FEATURE_FRAMES-1) and report onsets past it.
        self.assertGreaterEqual(len(new_onsets), 1)
        self.assertTrue(all(o >= CQT_FEATURE_FRAMES for o in new_onsets))
        self.assertTrue(any(abs(o - 250) < 50 for o in new_onsets))


# ---------------------------------------------------------------------------
# SegmentBuffer
# ---------------------------------------------------------------------------


class SegmentBufferTests(unittest.TestCase):
    def test_emits_full_window_after_target_frames(self):
        seg = SegmentBuffer(target_frames=CQT_FEATURE_FRAMES, min_onset_gap_ms=80)
        onsets = np.array([0])
        cols = np.zeros((CQT_FEATURE_BINS, 300), dtype=np.float32)
        windows = seg.push(cols, onsets)
        self.assertEqual(len(windows), 1)
        w = windows[0]
        self.assertEqual(w.source_frames, CQT_FEATURE_FRAMES)
        self.assertFalse(w.truncated)

    def test_emits_truncated_window_on_early_onset(self):
        # Use min_onset_gap_ms=0 so back-to-back onsets are not debounced.
        seg = SegmentBuffer(target_frames=CQT_FEATURE_FRAMES, min_onset_gap_ms=0)
        # 1) Onset at frame 0 starts COLLECTING.
        # 2) Append 60 columns (no full-window emit yet because 60 < 188).
        # 3) A new onset at frame 60 forces a truncated emit of those 60 frames.
        seg.push(np.ones((CQT_FEATURE_BINS, 60), dtype=np.float32), np.array([0]))
        windows = seg.push(np.ones((CQT_FEATURE_BINS, 0), dtype=np.float32), np.array([60]))
        self.assertEqual(len(windows), 1)
        w = windows[0]
        self.assertTrue(w.truncated)
        self.assertEqual(w.source_frames, 60)
        self.assertEqual(w.cqt.shape[1], 60)


# ---------------------------------------------------------------------------
# End-to-end pipeline (CPU only; no TF)
# ---------------------------------------------------------------------------


class EndToEndRunnerTests(unittest.TestCase):
    def test_message_schema_with_classifier_disabled(self):
        # With classifier off we should still get cqt_columns messages
        # (rate-limited to ~20 Hz) and the cqt_columns schema must
        # match what the WebSocket handler forwards to the browser.
        from app.pipeline.runner import PipelineRunner

        runner = PipelineRunner(with_classifier=False)
        rng = np.random.default_rng(2)
        sr = AUDIO_SAMPLE_RATE

        total = 6 * sr
        audio = rng.standard_normal(total).astype(np.float32) * 0.02
        for i in range(0, 6 * 2):
            start = int(i * 0.5 * sr)
            audio[start : start + 1024] += np.hanning(1024) * 0.3

        chord_messages = 0
        cqt_messages = 0
        for off in range(0, total, 4096):
            msgs = runner.ingest_pcm(audio[off : off + 4096])
            for m in msgs:
                self.assertIn(m["type"], {"chord", "cqt_columns", "error", "reset"})
                if m["type"] == "chord":
                    chord_messages += 1
                    self.assertGreaterEqual(
                        set(m.keys()),
                        {
                            "type",
                            "raw_label",
                            "display_label",
                            "confidence",
                            "predicted_index",
                            "onset_time",
                            "duration",
                            "truncated",
                            "source_frames",
                        },
                    )
                    self.assertIn(m["raw_label"], MODEL_LABELS)
                elif m["type"] == "cqt_columns":
                    cqt_messages += 1
                    self.assertIn("columns", m)
                    # The runner flattens n_bins * n_cols floats.
                    self.assertEqual(m["n_bins"] * m["n_cols"], len(m["columns"]))
                    self.assertEqual(m["n_bins"], CQT_FEATURE_BINS)
        runner.close()
        # Classifier is disabled so no chord messages expected.
        self.assertEqual(chord_messages, 0)
        # cqt_columns are rate-limited to ~20 Hz, so 6 s of audio
        # should produce a healthy number of them.
        self.assertGreaterEqual(cqt_messages, 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
