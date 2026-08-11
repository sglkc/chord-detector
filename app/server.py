"""FastAPI server for the real-time chord detection demo.

Run with::

    uv run uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
``GET  /``             - the single-page HTML shell.
``GET  /static/*``     - CSS / JS / AudioWorklet assets.
``WS   /ws``           - bidirectional WebSocket carrying Float32 PCM
                        audio in and JSON messages out.

Message protocol (server -> client)
-----------------------------------
``{"type": "ready"}``
    Sent on connect.

``{"type": "cqt_columns", "n_bins": int, "n_cols": int,
   "time_s": float, "columns": [float, ...]}``
    The most recent ``CQT_TRAIL_COLUMNS`` CQT columns, dB-scaled, in
    row-major (C) order. Re-shape on the client to
    ``(n_bins, n_cols)``.

``{"type": "onset", "column": int, "time_s": float}``
    A newly detected Superflux onset. ``column`` is the global CQT
    column index (same coordinate system as ``end_column``).

``{"type": "chord", "raw_label": str, "display_label": str,
   "confidence": float, "onset_time": float, "duration": float,
   "truncated": bool, "source_frames": int, "onset_column": int}``
    A completed chord classification. ``onset_column`` matches the
    Superflux onset that started the segment.

``{"type": "error", "message": str}``
    Something went wrong; the client should surface it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .pipeline.classifier import ChordClassifier
from .pipeline.runner import PipelineRunner


logger = logging.getLogger("chord-detection-demo")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Shared model (loaded once at startup)
# ---------------------------------------------------------------------------

_shared_classifier: Optional[ChordClassifier] = None
_model_status: dict = {
    "loaded": False,
    "load_time_s": 0.0,
    "error": None,
}
_predict_lock = threading.Lock()

# Reject binary frames larger than this (WS_CHUNK_SAMPLES * 4 bytes/float
# * 4x headroom). Protects against accidental huge uploads.
_MAX_WS_AUDIO_BYTES: int = config.WS_CHUNK_SAMPLES * 4 * 4

# Cap coalesced PCM depth under overload (~8 * 85 ms ≈ 0.7 s).
_MAX_PENDING_CHUNKS: int = 8


def _load_shared_classifier() -> None:
    """Load the process-wide ChordClassifier into module globals."""
    global _shared_classifier, _model_status
    t0 = time.monotonic()
    try:
        _shared_classifier = ChordClassifier(predict_lock=_predict_lock)
        _model_status = {
            "loaded": True,
            "load_time_s": time.monotonic() - t0,
            "error": None,
        }
        logger.info(
            "Shared classifier loaded in %.2fs from %s",
            _model_status["load_time_s"],
            config.MODEL_PATH,
        )
    except Exception as exc:  # noqa: BLE001 - surface any TF / IO error
        _shared_classifier = None
        _model_status = {
            "loaded": False,
            "load_time_s": time.monotonic() - t0,
            "error": f"{type(exc).__name__}: {exc}",
        }
        logger.exception("Failed to load shared classifier")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CNN once at startup; release on shutdown."""
    global _shared_classifier
    _load_shared_classifier()
    yield
    _shared_classifier = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Real-Time Chord Detection Demo",
    description=(
        "Browser microphone -> librosa CQT + Superflux onsets -> CNN "
        "chord classification, served over a single WebSocket."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Mount /static for CSS, JS, AudioWorklet.
app.mount(
    "/static",
    StaticFiles(directory=str(config.STATIC_DIR), check_dir=False),
    name="static",
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page HTML shell."""
    index_path: Path = config.TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h1>index.html missing</h1><p>Did you forget to create "
            "app/templates/index.html?</p>",
            status_code=500,
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/healthz")
async def healthz() -> dict:
    """Cheap health-check endpoint with real model-load status."""
    return {
        "ok": True,
        "model_loaded": bool(_model_status.get("loaded")),
        "model_error": _model_status.get("error"),
        "model_load_time_s": _model_status.get("load_time_s"),
        "sample_rate": config.AUDIO_SAMPLE_RATE,
        "cqt_bins": config.CQT_FEATURE_BINS,
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Per-connection pipeline (DSP state only; model is shared).

    Audio flows client -> server as binary WebSocket frames containing
    Float32 little-endian mono PCM. Messages flow server -> client as
    UTF-8 JSON text frames.

    PCM processing uses a single in-flight worker with coalescing.
    Control commands that mutate the runner (``reset``, ``set``) await
    that worker so they never race with ``ingest_pcm``.
    """
    await websocket.accept()
    runner: Optional[PipelineRunner] = None
    # Coalesce PCM that arrives while a previous ingest is still running.
    pending_chunks: List[np.ndarray] = []
    pcm_task: Optional[asyncio.Task] = None
    dropped_chunks: int = 0

    async def _flush_pcm() -> None:
        """Drain and process coalesced PCM until the queue is empty."""
        nonlocal pending_chunks
        assert runner is not None
        while pending_chunks:
            chunks = pending_chunks
            pending_chunks = []
            if len(chunks) == 1:
                samples = chunks[0]
            else:
                samples = np.concatenate(chunks)
            try:
                outbound = await asyncio.to_thread(runner.ingest_pcm, samples)
            except Exception as exc:  # pragma: no cover - safety net
                logger.exception("ingest_pcm failed")
                try:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": f"ingest: {exc}"})
                    )
                except Exception:
                    pass
                # Keep draining residual / newly queued PCM so an error
                # mid-stream does not stall the queue until the next frame.
                continue
            for msg in outbound:
                await websocket.send_text(json.dumps(msg))

    def _schedule_flush() -> None:
        nonlocal pcm_task
        if pcm_task is not None and not pcm_task.done():
            return
        pcm_task = asyncio.create_task(_flush_pcm())

    async def _await_runner_idle() -> None:
        """Block until no in-flight ingest is mutating the runner."""
        nonlocal pcm_task
        if pcm_task is not None and not pcm_task.done():
            try:
                await pcm_task
            except Exception:
                pass

    def _enqueue_pcm(samples: np.ndarray) -> None:
        """Append PCM, dropping oldest chunks if the queue is full."""
        nonlocal dropped_chunks
        pending_chunks.append(samples)
        while len(pending_chunks) > _MAX_PENDING_CHUNKS:
            pending_chunks.pop(0)
            dropped_chunks += 1

    try:
        # Always inject the shared slot (may be None if load failed) so
        # the runner never private-loads a second model per connection.
        runner = PipelineRunner(
            with_classifier=True,
            classifier=_shared_classifier,
            model_status=_model_status,
        )
        await websocket.send_text(json.dumps({"type": "ready"}))
        # Confirm model status on connect (page load already polls
        # /healthz; this keeps the pill in sync if status changed).
        await websocket.send_text(
            json.dumps({"type": "model_status", **runner.model_status})
        )
        logger.info(
            "WebSocket connected; pipeline ready. model_loaded=%s "
            "load_time_s=%.2f",
            runner.model_status["loaded"],
            runner.model_status["load_time_s"],
        )

        while True:
            message = await websocket.receive()

            # Disconnect / control frames.
            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is None:
                # A text frame from the client is treated as a
                # control command (e.g. "reset", "ping").
                text = message.get("text")
                if text is not None:
                    stripped = text.strip()
                    lower = stripped.lower()
                    # Serialize mutations with the PCM worker. Ping
                    # does not touch the runner and can proceed.
                    mutates = lower == "reset" or lower.startswith("set ")
                    if mutates:
                        await _await_runner_idle()
                        if lower == "reset":
                            # Discard audio that arrived during the
                            # previous session so it is not ingested
                            # after reset rebuilds DSP state.
                            pending_chunks.clear()
                    await _handle_control(websocket, runner, text)
                continue

            # Binary frame size guard.
            if len(data) > _MAX_WS_AUDIO_BYTES:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": (
                                f"Audio frame too large ({len(data)} bytes; "
                                f"max {_MAX_WS_AUDIO_BYTES})"
                            ),
                        }
                    )
                )
                continue

            # Binary frame: Float32 PCM.
            try:
                samples = np.frombuffer(data, dtype=np.float32).copy()
            except ValueError as exc:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": f"Bad audio frame: {exc!s}"})
                )
                continue

            # Coalesce into the pending queue (capped) and ensure a
            # single in-flight processor is running.
            prev_dropped = dropped_chunks
            _enqueue_pcm(samples)
            if dropped_chunks > prev_dropped:
                # Best-effort warning; do not spam more than once per
                # overload burst (only when we actually drop).
                try:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": (
                                    "PCM backpressure: dropped oldest audio "
                                    f"chunk(s); pending cap={_MAX_PENDING_CHUNKS}"
                                ),
                            }
                        )
                    )
                except Exception:
                    pass
            _schedule_flush()

        # Drain any in-flight work before teardown.
        if pcm_task is not None and not pcm_task.done():
            try:
                await pcm_task
            except Exception:
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client.")
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("WebSocket handler crashed")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        if pcm_task is not None and not pcm_task.done():
            pcm_task.cancel()
            try:
                await pcm_task
            except Exception:
                pass
        if runner is not None:
            runner.close()
        logger.info("Pipeline runner closed.")


async def _handle_control(websocket: WebSocket, runner: PipelineRunner, text: str) -> None:
    """Handle a small set of text-frame control commands.

    Supported commands
    ------------------
    ``ping``
        Cheap heartbeat; server replies with ``{"type": "pong"}``.
    ``reset``
        Drop in-flight pipeline state on the server side. The
        client also clears its canvas / chord card.
    ``set <key>=<value>``
        Update a runtime-tunable parameter on the pipeline. The
        key/value pair is forwarded to ``OnsetDetector.set_param``
        which validates the key against a whitelist. Example::

            set min_onset_gap_ms=300
            set peak_pick_delta=0.5

    Callers that mutate the runner must ensure no concurrent
    ``ingest_pcm`` is running (see websocket_endpoint).
    """
    cmd = text.strip()
    if not cmd:
        return
    lower = cmd.lower()
    if lower == "ping":
        await websocket.send_text(json.dumps({"type": "pong"}))
        return
    if lower == "reset":
        runner.reset()
        await websocket.send_text(json.dumps({"type": "ready"}))
        return
    if lower.startswith("set "):
        body = cmd[4:].strip()
        if "=" not in body:
            await websocket.send_text(
                json.dumps({"type": "error", "message": f"Bad set command: {cmd!r}"})
            )
            return
        key, value = body.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            applied = runner.onsets.set_param(key, value)
        except (KeyError, ValueError) as exc:
            await websocket.send_text(
                json.dumps({"type": "error", "message": f"set: {exc}"})
            )
            return
        await websocket.send_text(
            json.dumps(
                {
                    "type": "param_updated",
                    "key": key,
                    "value": applied,
                }
            )
        )
        return
    await websocket.send_text(
        json.dumps({"type": "error", "message": f"Unknown command: {cmd!r}"})
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """``python -m app.server`` entry point."""
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
