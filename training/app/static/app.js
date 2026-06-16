// app.js -- Real-time chord detection demo (frontend).
//
// Responsibilities:
//   1. Capture the microphone via getUserMedia + AudioWorklet.
//   2. Stream Float32 mono PCM to /ws as binary WebSocket frames.
//   3. Render incoming CQT columns on a scrolling canvas.
//   4. Display the latest chord label.
//
// Audio math lives entirely inside the worklet. The main thread is
// responsible for transport (WebSocket) and rendering only.

(() => {
  "use strict";

  // ----------------------------------------------------------------- //
  // DOM
  // ----------------------------------------------------------------- //

  const $ = (id) => document.getElementById(id);
  const chordEl = $("chord");
  const chordMetaEl = $("chord-meta");
  const cqtEl = $("cqt");
  const cqtMetaEl = $("cqt-meta");
  const startBtn = $("start");
  const resetBtn = $("reset");
  const statusEl = $("status");

  // ----------------------------------------------------------------- //
  // State
  // ----------------------------------------------------------------- //

  let audioCtx = null;
  let workletNode = null;
  let source = null;
  let mediaStream = null;
  let socket = null;
  let running = false;

  // CQT canvas state. We hold a rolling buffer of "trail" columns.
  // n_bins is learned from the first server message; default 216.
  let nBins = 216;
  // Trailing N CQT columns drawn on the canvas. Matches the server's
  // CQT_TRAIL_COLUMNS but is a pure client-side choice for the
  // scrolling window width.
  const TRAIL_COLS = 80;
  // Pixel dimensions of the canvas.
  const CANVAS_W = cqtEl.width;
  const CANVAS_H = cqtEl.height;
  // Pre-allocated column buffer (nBins x CANVAS_W); we copy incoming
  // columns into the rightmost slot and shift the rest left.
  let colBuffer = null;
  // ImageData of size CANVAS_W * CANVAS_H * 4.
  let imageData = null;
  // Pre-computed viridis-ish colormap. 256 RGB entries.
  let colormap = null;
  // dB range for mapping CQT values to color indices.
  const DB_MIN = -80.0;
  const DB_MAX = 0.0;

  // ----------------------------------------------------------------- //
  // Colormap (viridis approximation baked into JS)
  // ----------------------------------------------------------------- //

  function buildColormap() {
    // A simple 256-entry viridis approximation. Hand-tuned to be
    // visually distinct from black to bright yellow. Each entry is
    // an [r, g, b] triple in 0..255.
    const stops = [
      [ 13,   8, 135],   // dark purple
      [ 84,   2, 163],
      [139,  10, 165],
      [185,  50, 137],
      [219,  92, 104],
      [244, 136,  73],
      [254, 188,  43],
      [240, 249,  33],   // bright yellow
    ];
    const N = 256;
    const out = new Uint8Array(N * 3);
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1) * (stops.length - 1);
      const lo = Math.floor(t);
      const hi = Math.min(lo + 1, stops.length - 1);
      const f = t - lo;
      for (let c = 0; c < 3; c++) {
        out[i * 3 + c] = Math.round(stops[lo][c] + (stops[hi][c] - stops[lo][c]) * f);
      }
    }
    return out;
  }

  // ----------------------------------------------------------------- //
  // Canvas rendering
  // ----------------------------------------------------------------- //

  function ensureBuffers() {
    if (colormap === null) colormap = buildColormap();
    if (colBuffer === null || colBuffer.length !== nBins * CANVAS_W) {
      colBuffer = new Float32Array(nBins * CANVAS_W);
      colBuffer.fill(DB_MIN);
    }
    if (imageData === null) {
      imageData = new ImageData(CANVAS_W, CANVAS_H);
    }
  }

  // Shift the column buffer one column to the left and write the new
  // column at the right edge. colBuffer is laid out as a flat
  // CANVAS_W * nBins array in COLUMN-MAJOR order: column x occupies
  // indices [x*nBins, (x+1)*nBins). Shifting one column = advancing
  // the source pointer by nBins elements.
  function appendColumn(newCol) {
    // newCol: Float32Array of length nBins.
    if (newCol.length !== nBins) {
      // Server told us a different bin count; rebuild buffer.
      nBins = newCol.length;
      colBuffer = new Float32Array(nBins * CANVAS_W);
      colBuffer.fill(DB_MIN);
    }

    // Shift left by exactly ONE column (nBins elements).
    colBuffer.copyWithin(0, nBins);

    // Write the new column into the rightmost slot. The frequency
    // axis is BINS (rows). We want bin 0 (lowest frequency) at the
    // BOTTOM of the canvas and bin (nBins-1) at the TOP. The
    // browser's y axis grows downward, so we flip the bin index
    // when writing.
    const slotStart = (CANVAS_W - 1) * nBins;
    for (let i = 0; i < nBins; i++) {
      colBuffer[slotStart + (nBins - 1 - i)] = newCol[i];
    }
  }

  function drawCanvas() {
    if (colBuffer === null) return;
    const ctx = cqtEl.getContext("2d");
    const data = imageData.data;

    // For each canvas column, sample one CQT column from colBuffer
    // and map each bin -> a row in the image. Bin 0 (lowest
    // frequency) lives at the BOTTOM of the canvas (y = CANVAS_H-1)
    // and bin (nBins-1) (highest) at the TOP (y = 0). The buffer
    // is already stored in that flipped order by appendColumn, so
    // we map canvas-y to the raw bin index directly.
    for (let x = 0; x < CANVAS_W; x++) {
      const colStart = x * nBins;
      for (let y = 0; y < CANVAS_H; y++) {
        const binF = (y / (CANVAS_H - 1)) * (nBins - 1);
        const bin0 = Math.floor(binF);
        const bin1 = Math.min(bin0 + 1, nBins - 1);
        const f = binF - bin0;
        const v = colBuffer[colStart + bin0] * (1 - f) + colBuffer[colStart + bin1] * f;

        // Map dB to [0, 255] colormap index.
        const t = (v - DB_MIN) / (DB_MAX - DB_MIN);
        const idx = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;

        const px = (y * CANVAS_W + x) * 4;
        data[px]     = colormap[idx];
        data[px + 1] = colormap[idx + 1];
        data[px + 2] = colormap[idx + 2];
        data[px + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }

  // ----------------------------------------------------------------- //
  // WebSocket message handling
  // ----------------------------------------------------------------- //

  function handleServerMessage(msg) {
    if (msg.type === "ready") {
      setStatus("connected", "connected");
    } else if (msg.type === "model_status") {
      setModelStatus(msg);
    } else if (msg.type === "param_updated") {
      // Server confirmed a "set" command. Reflect it on the slider
      // so the UI agrees with the backend.
      const slider = document.querySelector(
        `[data-param-key="${msg.key}"]`
      );
      if (slider && document.activeElement !== slider) {
        slider.value = msg.value == null ? "" : String(msg.value);
        updateSliderLabel(slider);
      }
    } else if (msg.type === "cqt_columns") {
      ensureBuffers();
      // Re-shape the flat 1-D array into (n_cols, n_bins) and push
      // each column through appendColumn so the canvas scrolls.
      const nBinsServer = msg.n_bins;
      const nCols = msg.n_cols;
      if (nBinsServer !== nBins) {
        nBins = nBinsServer;
        colBuffer = new Float32Array(nBins * CANVAS_W);
        colBuffer.fill(DB_MIN);
      }
      const flat = msg.columns;
      // For each new column, copy its bins into a temporary array
      // and append. (nBins * nCols could be ~17k floats - well under
      // the WS message size limit.)
      const newCol = new Float32Array(nBins);
      for (let c = 0; c < nCols; c++) {
        for (let b = 0; b < nBins; b++) {
          newCol[b] = flat[c * nBins + b];
        }
        appendColumn(newCol);
      }
      drawCanvas();
      cqtMetaEl.textContent = `${nCols} new cols | time ${msg.time_s.toFixed(1)}s`;
    } else if (msg.type === "chord") {
      chordEl.textContent = msg.display_label;
      chordMetaEl.textContent =
        `confidence ${(msg.confidence * 100).toFixed(1)}%  |  ` +
        `onset ${msg.onset_time.toFixed(2)}s  |  ` +
        (msg.truncated ? `truncated (${msg.source_frames} frames)` : `full (188 frames)`);

      // Briefly flash the chord card so the user can see new
      // predictions.
      const card = document.querySelector(".chord-card");
      card.classList.remove("flash");
      // Force reflow so the animation restarts.
      // eslint-disable-next-line no-unused-expressions
      void card.offsetWidth;
      card.classList.add("flash");
      setTimeout(() => card.classList.remove("flash"), 220);
    } else if (msg.type === "error") {
      setStatus(`server: ${msg.message}`, "error");
    } else if (msg.type === "pong") {
      // Heartbeat response - nothing to do.
    }
  }

  function setStatus(text, klass) {
    statusEl.textContent = text;
    statusEl.classList.remove("connected", "error");
    if (klass) statusEl.classList.add(klass);
  }

  // ----------------------------------------------------------------- //
  // Model + button state helpers
  // ----------------------------------------------------------------- //

  const modelStatusEl = $("model-status");

  function setModelStatus(msg) {
    if (!modelStatusEl) return;
    if (msg.loaded) {
      const t = (msg.load_time_s || 0).toFixed(2);
      modelStatusEl.textContent = `model: ready (${t}s)`;
      modelStatusEl.classList.remove("loading", "error");
      modelStatusEl.classList.add("ready");
    } else if (msg.error) {
      modelStatusEl.textContent = `model: error — ${msg.error}`;
      modelStatusEl.classList.remove("loading", "ready");
      modelStatusEl.classList.add("error");
    } else {
      modelStatusEl.textContent = "model: loading…";
      modelStatusEl.classList.remove("ready", "error");
      modelStatusEl.classList.add("loading");
    }
  }

  function setStartButtonState(isRunning) {
    if (!startBtn) return;
    if (isRunning) {
      startBtn.textContent = "Stop microphone";
      startBtn.classList.add("danger");
    } else {
      startBtn.textContent = "Start microphone";
      startBtn.classList.remove("danger");
    }
  }

  function updateSliderLabel(slider) {
    const out = document.querySelector(
      `[data-param-label-for="${slider.dataset.paramKey}"]`
    );
    if (out) {
      out.textContent = slider.value === "" ? "default" : slider.value;
    }
  }

  // Push the current value of every .param-slider to the server.
  // Used on connect so reload-during-session doesn't lose tweaks.
  function sendAllParams() {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    document.querySelectorAll(".param-slider").forEach((slider) => {
      const v = slider.value;
      if (v === "" || v == null) return;
      socket.send(`set ${slider.dataset.paramKey}=${v}`);
    });
  }

  // ----------------------------------------------------------------- //
  // Audio capture
  // ----------------------------------------------------------------- //

  const WORKLET_SOURCE = `
class MicProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opt = (options && options.processorOptions) || {};
    this.chunkSize = opt.chunkSize || 4096;
    this.buffer = new Float32Array(this.chunkSize);
    this.bufferFill = 0;
    this.port.onmessage = (event) => { if (event.data === "stop") this.stopped = true; };
    this.stopped = false;
  }
  process(inputs) {
    if (this.stopped) return false;
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel || channel.length === 0) return true;
    let read = 0;
    while (read < channel.length) {
      const space = this.chunkSize - this.bufferFill;
      const toCopy = Math.min(space, channel.length - read);
      this.buffer.set(channel.subarray(read, read + toCopy), this.bufferFill);
      this.bufferFill += toCopy;
      read += toCopy;
      if (this.bufferFill === this.chunkSize) {
        const out = this.buffer;
        this.port.postMessage(out, [out.buffer]);
        this.buffer = new Float32Array(this.chunkSize);
        this.bufferFill = 0;
      }
    }
    return true;
  }
}
registerProcessor("mic-processor", MicProcessor);
`;

  async function ensureWorklet(audioContext) {
    // Register the worklet from an inline Blob URL so the server
    // doesn't need to serve worklet.js as a separate fetch.
    const blob = new Blob([WORKLET_SOURCE], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    try {
      await audioContext.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async function start() {
    if (running) return;
    running = true;
    setStartButtonState(true);
    resetBtn.disabled = false;
    setStatus("requesting microphone...", null);

    try {
      // 1. Open the WebSocket first so the server-side pipeline is
      //    ready by the time the first audio chunk arrives.
      const wsUrl = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
      socket = new WebSocket(wsUrl);
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        setStatus("socket open, requesting mic...", null);
        // Replay current slider values so a server-side reset
        // (e.g. "reset" command) doesn't silently drop the
        // user's tweaks.
        sendAllParams();
      };
      socket.onclose = () => {
        setStatus("disconnected", null);
        stop();
      };
      socket.onerror = () => setStatus("socket error", "error");
      socket.onmessage = (ev) => {
        try {
          handleServerMessage(JSON.parse(ev.data));
        } catch (err) {
          setStatus(`bad message: ${err.message}`, "error");
        }
      };

      // 2. Audio context at 48 kHz. Some browsers refuse and silently
      //    fall back; we trust the spec.
      audioCtx = new AudioContext({ sampleRate: 48000 });
      await ensureWorklet(audioCtx);

      // 3. Microphone. Request mono + no processing.
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });

      const src = audioCtx.createMediaStreamSource(mediaStream);
      source = src;
      workletNode = new AudioWorkletNode(audioCtx, "mic-processor", {
        processorOptions: { chunkSize: 4096 },
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      workletNode.port.onmessage = (ev) => {
        if (!socket || socket.readyState !== WebSocket.OPEN) return;
        // ev.data is a Float32Array; send its underlying ArrayBuffer
        // as a binary frame. No copy on send.
        const f32 = ev.data;
        socket.send(f32.buffer);
      };
      src.connect(workletNode);
      // Connect to a zero-gain destination so the worklet has a
      // valid graph node to process. We don't actually want audio
      // output.
      workletNode.connect(audioCtx.destination);

      setStatus("streaming", "connected");
    } catch (err) {
      setStatus(`error: ${err.message}`, "error");
      console.error(err);
      await stop();
    }
  }

  async function stop() {
    running = false;
    setStartButtonState(false);
    startBtn.disabled = false;
    if (workletNode) {
      try { workletNode.port.postMessage("stop"); } catch (_) { /* noop */ }
      workletNode.disconnect();
      workletNode = null;
    }
    if (source) source.disconnect();
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (audioCtx) {
      try { await audioCtx.close(); } catch (_) { /* noop */ }
      audioCtx = null;
    }
    if (socket) {
      try { socket.close(); } catch (_) { /* noop */ }
      socket = null;
    }
  }

  function reset() {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send("reset");
    }
    // Clear local canvas.
    if (colBuffer) colBuffer.fill(DB_MIN);
    if (imageData) {
      const ctx = cqtEl.getContext("2d");
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    }
    chordEl.textContent = "--";
    chordMetaEl.textContent = "awaiting audio\u2026";
  }

  // ----------------------------------------------------------------- //
  // Wire up buttons
  // ----------------------------------------------------------------- //

  startBtn.addEventListener("click", () => {
    if (running) {
      stop();
    } else {
      start();
    }
  });

  resetBtn.addEventListener("click", reset);

  // Initial canvas paint.
  const ctx0 = cqtEl.getContext("2d");
  ctx0.fillStyle = "#000";
  ctx0.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // ----------------------------------------------------------------- //
  // Settings panel: each slider has data-param-key="...". Live
  // values are sent to the server over the existing WebSocket
  // as ``set <key>=<value>`` text frames.
  // ----------------------------------------------------------------- //
  document.querySelectorAll(".param-slider").forEach((slider) => {
    updateSliderLabel(slider);
    slider.addEventListener("input", () => {
      updateSliderLabel(slider);
      if (socket && socket.readyState === WebSocket.OPEN) {
        const v = slider.value;
        socket.send(v === "" ? `set ${slider.dataset.paramKey}=none` : `set ${slider.dataset.paramKey}=${v}`);
      }
    });
  });
})();
