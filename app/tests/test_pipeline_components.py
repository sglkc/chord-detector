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
from app.pipeline.classifier import peak_normalize_db, stretch_features_to_frames
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
        # Fixed ref=1.0: values are finite dB; soft audio is typically
        # negative but we no longer require max exactly 0.
        self.assertTrue(np.isfinite(new_cols).all())
        self.assertLess(float(new_cols.max()), 40.0)

    def test_flatten_c_order_contract(self):
        """Server flattens trail as C-order (n_bins, n_cols)."""
        import librosa

        sr = AUDIO_SAMPLE_RATE
        rng = np.random.default_rng(3)
        y = rng.standard_normal(3 * sr).astype(np.float32) * 0.05
        stream = CQTStream(
            sample_rate=sr,
            fmin=librosa.note_to_hz("C1"),
            n_bins=CQT_FEATURE_BINS,
            bins_per_octave=CQT_BINS_PER_OCTAVE,
            hop_length=CQT_HOP_LENGTH,
            analysis_window_seconds=2.0,
        )
        stream.update(y, total_appended=len(y))
        trail = stream.columns[:, -min(80, stream.columns.shape[1]) :]
        flat = trail.astype(np.float32).flatten(order="C")
        n_bins, n_cols = trail.shape
        self.assertEqual(len(flat), n_bins * n_cols)
        # Index contract used by the browser: flat[b * n_cols + c].
        for b in (0, n_bins // 2, n_bins - 1):
            for c in (0, n_cols // 2, n_cols - 1):
                self.assertEqual(flat[b * n_cols + c], trail[b, c])


# ---------------------------------------------------------------------------
# OnsetDetector
# ---------------------------------------------------------------------------


class OnsetDetectorTests(unittest.TestCase):
    def test_envelope_at_reads_global_column(self):
        det = OnsetDetector()
        det.envelope = np.array([0.1, 0.4, 0.9], dtype=np.float32)
        det._envelope_start_frame = 10
        self.assertAlmostEqual(det.envelope_at(11), 0.4, places=5)
        self.assertIsNone(det.envelope_at(9))
        self.assertIsNone(det.envelope_at(13))
        det.reset()

    def test_peak_pick_delta_stays_on_unit_interval(self):
        det = OnsetDetector()
        self.assertAlmostEqual(det.set_param("peak_pick_delta", "0.2"), 0.2)
        self.assertAlmostEqual(det.peak_pick_params["delta"], 0.2)
        self.assertAlmostEqual(det.set_param("peak_pick_delta", "none"), 0.07)
        with self.assertRaises(ValueError):
            det.set_param("peak_pick_delta", "1.5")
        with self.assertRaises(ValueError):
            det.set_param("peak_pick_delta", "-0.1")
        det.reset()

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
            ref=1.0,
        )
        # Feed in chunks so the detector uses its trailing context path
        # (not a single giant update only).
        chunk = 32
        all_onsets = []
        for start in range(0, cqt.shape[1], chunk):
            new_onsets, _env = det.update(cqt[:, start : start + chunk])
            all_onsets.extend(int(o) for o in new_onsets)
        # The detector must ignore onsets in the warmup window
        # (frames 0..CQT_FEATURE_FRAMES-1) and report onsets past it.
        self.assertGreaterEqual(len(all_onsets), 1)
        self.assertTrue(all(o >= CQT_FEATURE_FRAMES for o in all_onsets))
        self.assertTrue(any(abs(o - 250) < 50 for o in all_onsets))

    def test_envelope_is_capped(self):
        det = OnsetDetector()
        # Push many columns in small chunks; envelope must stay bounded.
        cols = np.random.default_rng(0).standard_normal(
            (CQT_FEATURE_BINS, 64)
        ).astype(np.float32)
        for _ in range(40):
            det.update(cols)
        self.assertLessEqual(det.envelope.size, 512)


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

    def test_second_segment_after_truncate(self):
        """onset → columns → second onset → more columns → two windows."""
        seg = SegmentBuffer(target_frames=CQT_FEATURE_FRAMES, min_onset_gap_ms=0)
        # Onset at 0, then 40 columns (global 0..39).
        w1 = seg.push(np.full((CQT_FEATURE_BINS, 40), 1.0, dtype=np.float32), np.array([0]))
        self.assertEqual(len(w1), 0)
        # Second onset at 40 with 50 more columns (global 40..89).
        # First segment should truncate at 40 frames; second collects 50.
        w2 = seg.push(np.full((CQT_FEATURE_BINS, 50), 2.0, dtype=np.float32), np.array([40]))
        self.assertEqual(len(w2), 1)
        self.assertTrue(w2[0].truncated)
        self.assertEqual(w2[0].source_frames, 40)
        self.assertEqual(w2[0].onset_frame, 0)
        self.assertTrue(np.allclose(w2[0].cqt, 1.0))
        # Still collecting the second segment; force truncate with a
        # third onset and empty columns to emit it.
        w3 = seg.push(np.zeros((CQT_FEATURE_BINS, 0), dtype=np.float32), np.array([90]))
        self.assertEqual(len(w3), 1)
        self.assertTrue(w3[0].truncated)
        self.assertEqual(w3[0].source_frames, 50)
        self.assertEqual(w3[0].onset_frame, 40)
        self.assertTrue(np.allclose(w3[0].cqt, 2.0))

    def test_mid_batch_skips_pre_onset_columns(self):
        """Columns before the onset in the same batch must not be kept."""
        seg = SegmentBuffer(target_frames=CQT_FEATURE_FRAMES, min_onset_gap_ms=0)
        # 20 pre-onset columns (values 1) then onset at global 20 with
        # 30 post-onset columns (values 2) in one push of 50 cols.
        cols = np.concatenate(
            [
                np.full((CQT_FEATURE_BINS, 20), 1.0, dtype=np.float32),
                np.full((CQT_FEATURE_BINS, 30), 2.0, dtype=np.float32),
            ],
            axis=1,
        )
        windows = seg.push(cols, np.array([20]))
        self.assertEqual(len(windows), 0)
        # Finalize with a later onset.
        windows = seg.push(
            np.zeros((CQT_FEATURE_BINS, 0), dtype=np.float32), np.array([50])
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].source_frames, 30)
        self.assertTrue(np.allclose(windows[0].cqt, 2.0))
        self.assertEqual(windows[0].onset_frame, 20)

    def test_late_onset_splits_already_collected_columns(self):
        """Lagged onset inside prior batch range splits buffer, no lost frames."""
        seg = SegmentBuffer(target_frames=CQT_FEATURE_FRAMES, min_onset_gap_ms=0)
        # Onset 0, collect 80 columns (global 0..79), all value 1.
        w0 = seg.push(np.full((CQT_FEATURE_BINS, 80), 1.0, dtype=np.float32), np.array([0]))
        self.assertEqual(len(w0), 0)
        # Next batch is global 80..99 (value 2), but peak-pick reports a
        # late onset at frame 50 (already inside the previous buffer).
        windows = seg.push(
            np.full((CQT_FEATURE_BINS, 20), 2.0, dtype=np.float32), np.array([50])
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].source_frames, 50)
        self.assertEqual(windows[0].onset_frame, 0)
        self.assertTrue(np.allclose(windows[0].cqt, 1.0))
        # New segment was seeded with frames 50..79 (1.0) plus 80..99 (2.0).
        w2 = seg.push(np.zeros((CQT_FEATURE_BINS, 0), dtype=np.float32), np.array([100]))
        self.assertEqual(len(w2), 1)
        self.assertEqual(w2[0].onset_frame, 50)
        self.assertEqual(w2[0].source_frames, 50)  # 30 + 20
        self.assertTrue(np.allclose(w2[0].cqt[:, :30], 1.0))
        self.assertTrue(np.allclose(w2[0].cqt[:, 30:], 2.0))


# ---------------------------------------------------------------------------
# CNN input peak-norm (training is amplitude_to_db(..., ref=np.max))
# ---------------------------------------------------------------------------


class PeakNormalizeTests(unittest.TestCase):
    def test_peak_becomes_zero_and_offsets_preserved(self):
        x = np.array([[-20.0, -40.0], [-30.0, -10.0]], dtype=np.float32)
        y = peak_normalize_db(x)
        self.assertAlmostEqual(float(y.max()), 0.0)
        self.assertAlmostEqual(float(y[1, 1]), 0.0)
        self.assertAlmostEqual(float(y[0, 0]), -10.0)
        self.assertAlmostEqual(float(y[0, 1]), -30.0)

    def test_empty_passthrough(self):
        empty = np.zeros((216, 0), dtype=np.float32)
        out = peak_normalize_db(empty)
        self.assertEqual(out.shape, (216, 0))

    def test_stretch_then_norm_keeps_188_frames(self):
        short = np.linspace(-60.0, -20.0, 40, dtype=np.float32)
        window = np.stack([short] * CQT_FEATURE_BINS, axis=0)
        stretched = stretch_features_to_frames(window, CQT_FEATURE_FRAMES)
        normed = peak_normalize_db(stretched)
        self.assertEqual(normed.shape, (CQT_FEATURE_BINS, CQT_FEATURE_FRAMES))
        self.assertAlmostEqual(float(normed.max()), 0.0, places=4)


# ---------------------------------------------------------------------------
# Runner classifier injection
# ---------------------------------------------------------------------------


class PipelineRunnerInjectionTests(unittest.TestCase):
    def test_explicit_none_classifier_does_not_private_load(self):
        """Server-style injection of None must not load a private model."""
        from app.pipeline.runner import PipelineRunner

        status = {"loaded": False, "load_time_s": 1.25, "error": "boom"}
        runner = PipelineRunner(
            with_classifier=True,
            classifier=None,
            model_status=status,
        )
        self.assertIsNone(runner.classifier)
        self.assertFalse(runner.model_status["loaded"])
        self.assertEqual(runner.model_status["error"], "boom")
        runner.close()


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
        onset_messages = 0
        for off in range(0, total, 4096):
            msgs = runner.ingest_pcm(audio[off : off + 4096])
            for m in msgs:
                self.assertIn(m["type"], {"chord", "cqt_columns", "onset", "error", "reset"})
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
                            "onset_column",
                        },
                    )
                    self.assertIn(m["raw_label"], MODEL_LABELS)
                elif m["type"] == "cqt_columns":
                    cqt_messages += 1
                    self.assertIn("columns", m)
                    # The runner flattens n_bins * n_cols floats (C-order).
                    self.assertEqual(m["n_bins"] * m["n_cols"], len(m["columns"]))
                    self.assertEqual(m["n_bins"], CQT_FEATURE_BINS)
                    self.assertIn("end_column", m)
                    self.assertGreaterEqual(m["end_column"], m["n_cols"])
                elif m["type"] == "onset":
                    onset_messages += 1
                    self.assertIn("column", m)
                    self.assertIn("time_s", m)
                    self.assertIn("strength", m)
                    self.assertIsInstance(m["column"], int)
                    self.assertGreaterEqual(m["column"], 0)
                    self.assertGreaterEqual(m["time_s"], 0.0)
        runner.close()
        # Classifier is disabled so no chord messages expected.
        self.assertEqual(chord_messages, 0)
        # Synthetic clicks after the Superflux warmup should produce
        # at least one onset marker for the spectrogram overlay.
        self.assertGreaterEqual(onset_messages, 1)
        # cqt_columns are rate-limited to ~20 Hz, so 6 s of audio
        # should produce a healthy number of them.
        self.assertGreaterEqual(cqt_messages, 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
