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

``{"type": "chord", "raw_label": str, "display_label": str,
   "confidence": float, "onset_time": float, "duration": float,
   "truncated": bool, "source_frames": int}``
    A completed chord classification.

``{"type": "error", "message": str}``
    Something went wrong; the client should surface it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .pipeline.runner import PipelineRunner


logger = logging.getLogger("chord-detection-demo")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


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
    """Cheap health-check endpoint."""
    return {
        "ok": True,
        "model_loaded": True,  # we only get here if the import graph succeeded
        "sample_rate": config.AUDIO_SAMPLE_RATE,
        "cqt_bins": config.CQT_FEATURE_BINS,
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Per-connection pipeline.

    Audio flows client -> server as binary WebSocket frames containing
    Float32 little-endian mono PCM. Messages flow server -> client as
    UTF-8 JSON text frames.
    """
    await websocket.accept()
    runner: Optional[PipelineRunner] = None
    try:
        runner = PipelineRunner(with_classifier=True)
        await websocket.send_text(json.dumps({"type": "ready"}))
        logger.info("WebSocket connected; pipeline ready.")

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
                    await _handle_control(websocket, runner, text)
                continue

            # Binary frame: Float32 PCM.
            try:
                samples = np.frombuffer(data, dtype=np.float32)
            except ValueError as exc:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": f"Bad audio frame: {exc!s}"})
                )
                continue

            # All blocking DSP / TF calls happen in a thread so we
            # don't stall the asyncio event loop.
            outbound = await asyncio.to_thread(runner.ingest_pcm, samples)

            for msg in outbound:
                await websocket.send_text(json.dumps(msg))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client.")
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("WebSocket handler crashed")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        if runner is not None:
            runner.close()
        logger.info("Pipeline runner closed.")


async def _handle_control(websocket: WebSocket, runner: PipelineRunner, text: str) -> None:
    """Handle a small set of text-frame control commands."""
    cmd = text.strip().lower()
    if cmd == "ping":
        await websocket.send_text(json.dumps({"type": "pong"}))
    elif cmd == "reset":
        runner.reset()
        await websocket.send_text(json.dumps({"type": "ready"}))
    else:
        await websocket.send_text(
            json.dumps({"type": "error", "message": f"Unknown command: {cmd!r}"})
        )


# Add reset() to PipelineRunner for the "reset" command.
def _patch_runner_reset() -> None:
    """Add a no-arg reset() to PipelineRunner that drops in-flight state."""

    def reset(self: PipelineRunner) -> None:  # type: ignore[override]
        self.segments.reset()
        self.onsets.reset()
        self.cqt = CQTStream()
        self.buffer = AudioRingBuffer()
        self._start_time = None
        self._last_cqt_emit_ms = 0.0

    PipelineRunner.reset = reset  # type: ignore[attr-defined]


_patch_runner_reset()


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
