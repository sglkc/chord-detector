"""End-to-end tests for the live FastAPI demo server.

Boots a real uvicorn process, then checks HTTP routes and the WebSocket
audio pipeline without a browser. Run from the repository root::

    .venv/bin/python -m app.tests.test_e2e_server
    # or
    .venv/bin/python -m unittest app.tests.test_e2e_server -v

Requires the model symlink (or weights) at ``app/models/*.keras``.
Optional chord classification uses a local validation clip under
``training/validations/`` when present (gitignored datasets).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.config import (
    AUDIO_SAMPLE_RATE,
    CQT_FEATURE_BINS,
    CQT_TRAIL_COLUMNS,
    MODEL_LABELS,
    MODEL_PATH,
    WS_CHUNK_SAMPLES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Local research artifact; not required for the core HTTP/CQT e2e path.
_CHORD_SAMPLE = (
    REPO_ROOT
    / "training"
    / "validations"
    / "normal_sustain"
    / "normal_sustain-1.ogg"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes, str]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
        return int(response.status), body, content_type


class LiveServerE2ETests(unittest.TestCase):
    """Process-level e2e: real uvicorn + HTTP + WebSocket."""

    process: Optional[subprocess.Popen] = None
    base_url: str = ""
    ws_url: str = ""
    port: int = 0

    @classmethod
    def setUpClass(cls) -> None:
        if not MODEL_PATH.exists():
            raise unittest.SkipTest(
                f"Model missing at {MODEL_PATH}. Create the app/models "
                "symlink (see app/README.md) before running e2e tests."
            )

        cls.port = _free_port()
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.ws_url = f"ws://127.0.0.1:{cls.port}/ws"

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Prefer the repo venv interpreter so TF / FastAPI resolve correctly.
        python = sys.executable
        cls.process = subprocess.Popen(
            [
                python,
                "-m",
                "uvicorn",
                "app.server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(cls.port),
                "--log-level",
                "warning",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        cls._wait_until_ready(timeout_s=120.0)

    @classmethod
    def tearDownClass(cls) -> None:
        proc = cls.process
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass
        cls.process = None

    @classmethod
    def _wait_until_ready(cls, timeout_s: float) -> None:
        assert cls.process is not None
        deadline = time.monotonic() + timeout_s
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                output = cls.process.stdout.read() if cls.process.stdout else ""
                raise RuntimeError(
                    f"uvicorn exited early (code={cls.process.returncode}).\n{output}"
                )
            try:
                payload = _http_json(f"{cls.base_url}/healthz", timeout=2.0)
                if payload.get("ok") and payload.get("model_loaded"):
                    return
                last_error = RuntimeError(f"healthz not ready: {payload}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            time.sleep(0.4)

        output = ""
        if cls.process.stdout is not None:
            # Non-blocking drain is hard; terminate will still capture nothing
            # if still running — surface whatever we know.
            pass
        cls.tearDownClass()
        raise TimeoutError(
            f"Server did not become ready within {timeout_s:.0f}s "
            f"(last error: {last_error})"
        )

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def test_healthz_reports_model_loaded(self) -> None:
        payload = _http_json(f"{self.base_url}/healthz")
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["model_loaded"], msg=payload)
        self.assertIsNone(payload["model_error"])
        self.assertEqual(payload["sample_rate"], AUDIO_SAMPLE_RATE)
        self.assertEqual(payload["cqt_bins"], CQT_FEATURE_BINS)
        self.assertGreater(float(payload["model_load_time_s"]), 0.0)

    def test_index_and_static_assets(self) -> None:
        status, body, content_type = _http_get(f"{self.base_url}/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        html = body.decode("utf-8")
        self.assertIn("Chord Detection", html)
        self.assertIn("chord-list", html)
        self.assertIn("Download audio", html)
        self.assertIn("Upload audio", html)
        self.assertIn("/static/app.js", html)

        for path, needle in (
            ("/static/app.js", b"AudioContext"),
            ("/static/styles.css", b"{"),
            ("/static/worklet.js", b"AudioWorkletProcessor"),
        ):
            st, asset, _ = _http_get(f"{self.base_url}{path}")
            self.assertEqual(st, 200, msg=path)
            self.assertGreater(len(asset), 100, msg=path)
            self.assertIn(needle, asset)

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    def test_websocket_handshake_ping_and_cqt_stream(self) -> None:
        counts = asyncio.run(self._ws_cqt_session())
        self.assertGreaterEqual(counts["ready"], 1)
        self.assertGreaterEqual(counts["model_status_loaded"], 1)
        self.assertGreaterEqual(counts["pong"], 1)
        self.assertGreaterEqual(counts["cqt_columns"], 1)
        self.assertEqual(counts["errors"], 0)

    def test_websocket_classifies_validation_audio(self) -> None:
        if not _CHORD_SAMPLE.is_file():
            self.skipTest(f"validation clip not present: {_CHORD_SAMPLE}")
        result = asyncio.run(self._ws_chord_session(_CHORD_SAMPLE))
        self.assertGreaterEqual(
            result["cqt_columns"],
            1,
            msg="expected spectrogram updates while streaming audio",
        )
        self.assertGreaterEqual(
            result["chord"],
            1,
            msg="expected at least one chord classification",
        )
        chord = result["first_chord"]
        assert chord is not None
        for key in (
            "raw_label",
            "display_label",
            "confidence",
            "predicted_index",
            "onset_time",
            "duration",
            "truncated",
            "source_frames",
            "onset_column",
            "strength",
        ):
            self.assertIn(key, chord)
        self.assertIn(chord["raw_label"], MODEL_LABELS)
        self.assertGreaterEqual(float(chord["confidence"]), 0.0)
        self.assertLessEqual(float(chord["confidence"]), 1.0)

    async def _ws_cqt_session(self) -> Dict[str, int]:
        import websockets

        counts = {
            "ready": 0,
            "model_status_loaded": 0,
            "pong": 0,
            "cqt_columns": 0,
            "errors": 0,
        }
        async with websockets.connect(self.ws_url, open_timeout=15) as ws:
            for _ in range(2):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") == "ready":
                    counts["ready"] += 1
                elif msg.get("type") == "model_status":
                    if msg.get("loaded"):
                        counts["model_status_loaded"] += 1
                    else:
                        self.fail(f"model_status not loaded: {msg}")

            await ws.send("ping")
            pong = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(pong.get("type"), "pong")
            counts["pong"] += 1

            # ~2.5 s of low-level noise + a click so CQT/onset stages engage.
            sr = AUDIO_SAMPLE_RATE
            total = int(2.5 * sr)
            rng = np.random.default_rng(0)
            audio = (rng.standard_normal(total).astype(np.float32) * 0.01)
            click_at = int(1.0 * sr)
            audio[click_at : click_at + 128] += np.hanning(128).astype(np.float32) * 0.4

            await self._stream_pcm(ws, audio, counts)

        return counts

    async def _ws_chord_session(self, sample_path: Path) -> Dict[str, Any]:
        import librosa
        import websockets

        y, _ = librosa.load(str(sample_path), sr=AUDIO_SAMPLE_RATE, mono=True)
        audio = y.astype(np.float32)

        counts: Dict[str, Any] = {
            "cqt_columns": 0,
            "chord": 0,
            "errors": 0,
            "first_chord": None,
        }
        async with websockets.connect(self.ws_url, open_timeout=15) as ws:
            # Drain handshake.
            for _ in range(2):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") == "model_status" and not msg.get("loaded"):
                    self.fail(f"model not loaded on connect: {msg}")

            await self._stream_pcm(ws, audio, counts, final_drain_s=2.0)

        return counts

    async def _stream_pcm(
        self,
        ws: Any,
        audio: np.ndarray,
        counts: Dict[str, Any],
        final_drain_s: float = 1.0,
    ) -> None:
        chunk = WS_CHUNK_SAMPLES
        for offset in range(0, len(audio), chunk):
            frame = audio[offset : offset + chunk]
            if len(frame) < chunk:
                frame = np.pad(frame, (0, chunk - len(frame)))
            await ws.send(frame.astype(np.float32, copy=False).tobytes())
            await self._drain(ws, counts, timeout_s=0.05)

        await self._drain(ws, counts, timeout_s=final_drain_s)

    async def _drain(
        self,
        ws: Any,
        counts: Dict[str, Any],
        timeout_s: float,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if not isinstance(raw, str):
                continue
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "cqt_columns":
                counts["cqt_columns"] = int(counts.get("cqt_columns", 0)) + 1
                self.assertEqual(msg["n_bins"], CQT_FEATURE_BINS)
                self.assertEqual(msg["n_cols"], CQT_TRAIL_COLUMNS)
                self.assertEqual(len(msg["columns"]), CQT_FEATURE_BINS * CQT_TRAIL_COLUMNS)
            elif kind == "chord":
                counts["chord"] = int(counts.get("chord", 0)) + 1
                if counts.get("first_chord") is None:
                    counts["first_chord"] = msg
            elif kind == "error":
                counts["errors"] = int(counts.get("errors", 0)) + 1
            elif kind == "pong":
                counts["pong"] = int(counts.get("pong", 0)) + 1


if __name__ == "__main__":
    unittest.main()
