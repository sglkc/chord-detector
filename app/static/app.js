// app.js -- Real-time chord detection demo (frontend).
//
// Responsibilities:
//   1. Capture the microphone via getUserMedia + AudioWorklet.
//   2. Stream Float32 mono PCM to /ws as binary WebSocket frames.
//   3. Render incoming CQT columns on a scrolling canvas.
//   4. Overlay Superflux onsets as red vertical lines with the
//      classified chord for that segment (rotated if it would clip).
//   5. Display the latest chord label.
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
  const cqtLabelsEl = $("cqt-labels");
  const cqtMetaEl = $("cqt-meta");
  const startBtn = $("start");
  const resetBtn = $("reset");
  const statusEl = $("status");
  const themeToggle = $("theme-toggle");

  // ----------------------------------------------------------------- //
  // State
  // ----------------------------------------------------------------- //

  let audioCtx = null;
  let workletNode = null;
  let silentGain = null;
  let source = null;
  let mediaStream = null;
  let socket = null;
  let running = false;

  // CQT canvas state. We hold a rolling buffer of canvas-width columns.
  // n_bins is learned from the first server message; default 216.
  let nBins = 216;
  // Pixel dimensions of the canvas.
  const CANVAS_W = cqtEl.width;
  const CANVAS_H = cqtEl.height;
  // Pre-allocated column buffer (nBins x CANVAS_W); laid out
  // COLUMN-MAJOR: column x occupies [x*nBins, (x+1)*nBins).
  let colBuffer = null;
  // ImageData of size CANVAS_W * CANVAS_H * 4.
  let imageData = null;
  // Pre-computed viridis-ish colormap. 256 RGB entries.
  let colormap = null;
  // dB range for mapping CQT values to color indices.
  const DB_MIN = -80.0;
  const DB_MAX = 0.0;
  // Last server ``end_column`` we painted. The server sends a short trail
  // snapshot each tick; we only scroll by (end_column - lastEndColumn)
  // new columns so history fills the full canvas width over time.
  let lastEndColumn = -1;
  // Superflux onsets on the scrolling canvas. Each entry is
  // { column, label, confidence, endColumn }. label/confidence are
  // filled in when the CNN classifies the segment that started here.
  // Pruned once the onset scrolls off the left edge.
  let onsets = [];

  const REQUIRED_SAMPLE_RATE = 48000;

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

  // Write one CQT column (length nBins, low-frequency first) into
  // colBuffer at canvas column x. Frequency axis is flipped so bin 0
  // (lowest) sits at the BOTTOM of the canvas.
  function writeColumnAt(x, newCol) {
    const slotStart = x * nBins;
    for (let i = 0; i < nBins; i++) {
      colBuffer[slotStart + (nBins - 1 - i)] = newCol[i];
    }
  }

  // Shift the rolling canvas buffer left by ``n`` columns and write
  // ``n`` new columns (length nBins each) into the rightmost slots.
  function appendColumns(newCols) {
    // newCols: Array of Float32Array, each length nBins, oldest first.
    const n = newCols.length;
    if (n <= 0) return;
    if (n >= CANVAS_W) {
      // Only the newest CANVAS_W columns fit; replace the whole buffer.
      const start = n - CANVAS_W;
      for (let i = 0; i < CANVAS_W; i++) {
        writeColumnAt(i, newCols[start + i]);
      }
      return;
    }
    // Shift existing content left by n columns (nBins elements each).
    colBuffer.copyWithin(0, n * nBins);
    for (let i = 0; i < n; i++) {
      writeColumnAt(CANVAS_W - n + i, newCols[i]);
    }
  }

  // Apply a server trail snapshot. The payload is always the latest
  // CQT_TRAIL_COLUMNS (full snapshot, not a delta). We use end_column
  // to advance the scrolling history by only the newly produced columns
  // so the canvas fills left-to-right over time instead of showing a
  // thin strip on the right edge forever.
  function applyTrailSnapshot(flat, nCols, nBinsServer, endColumn) {
    if (nBinsServer !== nBins) {
      nBins = nBinsServer;
      colBuffer = new Float32Array(nBins * CANVAS_W);
      colBuffer.fill(DB_MIN);
      lastEndColumn = -1;
      onsets = [];
    }
    ensureBuffers();

    if (nCols <= 0 || flat == null || flat.length === 0) return;

    // How many columns are new since the last paint?
    let nNew;
    if (lastEndColumn < 0 || endColumn < lastEndColumn) {
      // First paint after connect/reset, or stream rewound: seed with
      // the entire trail (fills the right edge; further ticks scroll).
      nNew = nCols;
    } else if (endColumn === lastEndColumn) {
      return; // nothing new
    } else {
      nNew = endColumn - lastEndColumn;
    }
    // We only have ``nCols`` values in the payload.
    nNew = Math.min(nNew, nCols, CANVAS_W);
    if (nNew <= 0) return;

    // Server flattens C-order (n_bins, n_cols): index = b * n_cols + c.
    // Take the rightmost nNew columns of the trail (the newest ones).
    const cols = new Array(nNew);
    for (let i = 0; i < nNew; i++) {
      const c = nCols - nNew + i;
      const newCol = new Float32Array(nBins);
      for (let b = 0; b < nBins; b++) {
        newCol[b] = flat[b * nCols + c];
      }
      cols[i] = newCol;
    }
    appendColumns(cols);
    lastEndColumn = endColumn;
  }

  function drawCanvas() {
    if (colBuffer === null) return;
    const ctx = cqtEl.getContext("2d");
    const data = imageData.data;

    // For each canvas column, sample one CQT column from colBuffer
    // and map each bin -> a row in the image. Bin 0 (lowest
    // frequency) lives at the BOTTOM of the canvas (y = CANVAS_H-1)
    // and bin (nBins-1) (highest) at the TOP (y = 0). The buffer
    // is already stored in that flipped order by writeColumnAt, so
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
    drawOnsetLines(ctx);
    drawOnsetLabels();
  }

  // Map a global CQT column index to a canvas x. Rightmost pixel is
  // lastEndColumn - 1; leftmost is lastEndColumn - CANVAS_W.
  function onsetColumnToX(column) {
    return column - lastEndColumn + CANVAS_W;
  }

  function recordOnset(column) {
    if (typeof column !== "number" || !Number.isFinite(column)) return;
    const col = Math.round(column);
    for (let i = 0; i < onsets.length; i++) {
      if (onsets[i].column === col) return;
    }
    onsets.push({
      column: col,
      label: null,
      confidence: null,
      endColumn: null,
    });
    onsets.sort((a, b) => a.column - b.column);
  }

  function attachChordLabel(column, label, confidence, sourceFrames) {
    if (!label) return;
    const col = typeof column === "number" && Number.isFinite(column)
      ? Math.round(column)
      : null;
    const endColumn =
      col != null && typeof sourceFrames === "number" && sourceFrames > 0
        ? col + Math.round(sourceFrames)
        : null;

    let best = null;
    let bestDist = Infinity;
    if (col != null) {
      for (let i = 0; i < onsets.length; i++) {
        const d = Math.abs(onsets[i].column - col);
        if (d < bestDist) {
          bestDist = d;
          best = onsets[i];
        }
      }
    }
    // Peak-pick / debounce can land a few frames off the segment start.
    if (best && bestDist <= 4) {
      best.label = label;
      best.confidence = confidence;
      if (endColumn != null) best.endColumn = endColumn;
      return;
    }
    if (col == null) return;
    onsets.push({
      column: col,
      label,
      confidence,
      endColumn,
    });
    onsets.sort((a, b) => a.column - b.column);
  }

  function pruneOnsets() {
    if (lastEndColumn < 0 || onsets.length === 0) return;
    const leftEdge = lastEndColumn - CANVAS_W;
    // Keep onsets still on-canvas or just off the right (not yet
    // painted into the spectrogram, but already detected).
    let i = 0;
    while (i < onsets.length && onsets[i].column < leftEdge) i += 1;
    if (i > 0) onsets = onsets.slice(i);
  }

  function drawOnsetLines(ctx) {
    if (lastEndColumn < 0 || onsets.length === 0) return;
    pruneOnsets();
    ctx.fillStyle = "#ff2020";
    for (let i = 0; i < onsets.length; i++) {
      const x = onsetColumnToX(onsets[i].column);
      if (x < 0 || x >= CANVAS_W) continue;
      // 2px so the marker stays visible when the canvas is CSS-scaled.
      ctx.fillRect(Math.round(x), 0, 2, CANVAS_H);
    }
  }

  function clearOnsetLabels() {
    if (!cqtLabelsEl) return;
    const ctx = cqtLabelsEl.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cqtLabelsEl.width, cqtLabelsEl.height);
  }

  // Crisp overlay (not pixelated like the spectrogram). Each label
  // belongs to the segment that starts at that onset: it sits in the
  // span until the next onset / segment end. Horizontal when that
  // span is wide enough; otherwise rotated 90° along the red line.
  function drawOnsetLabels() {
    if (!cqtLabelsEl) return;
    const cssW = cqtEl.clientWidth || CANVAS_W;
    const cssH = cqtEl.clientHeight || CANVAS_H;
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.max(1, Math.round(cssW * dpr));
    const bh = Math.max(1, Math.round(cssH * dpr));
    if (cqtLabelsEl.width !== bw || cqtLabelsEl.height !== bh) {
      cqtLabelsEl.width = bw;
      cqtLabelsEl.height = bh;
    }
    const ctx = cqtLabelsEl.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    if (lastEndColumn < 0 || onsets.length === 0) return;
    pruneOnsets();

    const sx = cssW / CANVAS_W;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";

    for (let i = 0; i < onsets.length; i++) {
      const o = onsets[i];
      if (!o.label) continue;
      const xPx = onsetColumnToX(o.column) * sx;
      if (xPx < -12 || xPx > cssW + 12) continue;

      let rightLimit = cssW;
      if (i + 1 < onsets.length) {
        rightLimit = Math.min(rightLimit, onsetColumnToX(onsets[i + 1].column) * sx);
      }
      if (typeof o.endColumn === "number") {
        rightLimit = Math.min(rightLimit, onsetColumnToX(o.endColumn) * sx);
      }

      const conf =
        typeof o.confidence === "number"
          ? `${Math.round(o.confidence * 100)}%`
          : "";
      const text = conf ? `${o.label} ${conf}` : o.label;
      const tw = ctx.measureText(text).width;
      const pad = 4;
      const avail = rightLimit - xPx;
      const fitsH = avail >= tw + pad + 2 && xPx + pad + tw <= cssW - 2 && xPx >= 0;

      const alpha =
        typeof o.confidence === "number"
          ? 0.5 + 0.5 * Math.max(0, Math.min(1, o.confidence))
          : 0.9;

      ctx.save();
      ctx.strokeStyle = "rgba(0,0,0,0.82)";
      ctx.fillStyle = `rgba(255,236,236,${alpha})`;

      if (fitsH) {
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.strokeText(text, xPx + pad, 4);
        ctx.fillText(text, xPx + pad, 4);
      } else {
        // 90° clockwise: reads top-to-bottom along the onset line.
        // Sit just to the right of the line, or to the left if the
        // right side would clip the canvas edge.
        const fontH = 12;
        const onRight = xPx + fontH + 2 < cssW;
        ctx.translate(onRight ? xPx + 5 : xPx - 2, 5);
        ctx.rotate(Math.PI / 2);
        ctx.textAlign = "left";
        ctx.textBaseline = onRight ? "bottom" : "top";
        ctx.strokeText(text, 0, 0);
        ctx.fillText(text, 0, 0);
      }
      ctx.restore();
    }
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
        if (msg.value == null) {
          // Cleared override — show as default (0 on the delta slider).
          slider.value = slider.dataset.skipInitial === "true" ? "0" : "";
        } else {
          slider.value = String(msg.value);
        }
        updateSliderLabel(slider);
      }
    } else if (msg.type === "cqt_columns") {
      // Trail is a short snapshot; end_column drives how far to scroll.
      applyTrailSnapshot(
        msg.columns,
        msg.n_cols,
        msg.n_bins,
        typeof msg.end_column === "number" ? msg.end_column : -1
      );
      drawCanvas();
      const endLabel =
        typeof msg.end_column === "number" ? ` | col ${msg.end_column}` : "";
      cqtMetaEl.textContent =
        `${msg.n_cols} trail | canvas ${CANVAS_W}px${endLabel} | time ${msg.time_s.toFixed(1)}s`;
    } else if (msg.type === "onset") {
      recordOnset(msg.column);
      drawCanvas();
    } else if (msg.type === "chord") {
      chordEl.textContent = msg.display_label;
      chordMetaEl.textContent =
        `confidence ${(msg.confidence * 100).toFixed(1)}%  |  ` +
        `onset ${msg.onset_time.toFixed(2)}s  |  ` +
        (msg.truncated ? `truncated (${msg.source_frames} frames)` : `full (188 frames)`);

      // Pin the short label to the onset that started this segment
      // so the spectrogram keeps a history (the card only shows the
      // latest prediction, which is often C:min on background noise).
      const onsetCol =
        typeof msg.onset_column === "number"
          ? msg.onset_column
          : typeof msg.onset_time === "number"
            ? Math.round(msg.onset_time * REQUIRED_SAMPLE_RATE / 512)
            : null;
      attachChordLabel(
        onsetCol,
        msg.display_label,
        msg.confidence,
        msg.source_frames
      );
      drawOnsetLabels();

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
      // Not loaded and no error yet — only true during server startup.
      modelStatusEl.textContent = "model: loading…";
      modelStatusEl.classList.remove("ready", "error");
      modelStatusEl.classList.add("loading");
    }
  }

  // Model is loaded once at server startup (FastAPI lifespan), not when
  // the user clicks Start. Fetch /healthz on page load so the pill
  // reflects real status without requiring a WebSocket / mic session.
  // The WS still re-sends model_status on connect as a confirmation.
  async function refreshModelStatusFromHealthz() {
    try {
      const res = await fetch("/healthz", { cache: "no-store" });
      if (!res.ok) {
        setModelStatus({ loaded: false, error: `HTTP ${res.status}` });
        return;
      }
      const h = await res.json();
      setModelStatus({
        loaded: !!h.model_loaded,
        load_time_s: h.model_load_time_s,
        error: h.model_error || null,
      });
    } catch (err) {
      setModelStatus({
        loaded: false,
        error: err && err.message ? err.message : "unreachable",
      });
    }
  }
  refreshModelStatusFromHealthz();

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
    if (!out) return;
    // Sliders marked skip-initial treat value 0 as "unset / default"
    // (peak_pick_delta starts at 0 without forcing delta=0 on the server).
    if (slider.dataset.skipInitial === "true" && (slider.value === "0" || slider.value === "")) {
      out.textContent = "default";
    } else {
      out.textContent = slider.value === "" ? "default" : slider.value;
    }
  }

  // Push the current value of every .param-slider to the server.
  // Used on connect so reload-during-session doesn't lose tweaks.
  // Sliders with data-skip-initial="true" are omitted until the user
  // moves them (so peak_pick_delta does not force delta=0 on connect).
  function sendAllParams() {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    document.querySelectorAll(".param-slider").forEach((slider) => {
      if (slider.dataset.skipInitial === "true") return;
      const v = slider.value;
      if (v === "" || v == null) return;
      socket.send(`set ${slider.dataset.paramKey}=${v}`);
    });
  }

  // ----------------------------------------------------------------- //
  // Audio capture
  // ----------------------------------------------------------------- //

  async function ensureWorklet(audioContext) {
    // Single source of truth: static/worklet.js (served by FastAPI).
    await audioContext.audioWorklet.addModule("/static/worklet.js");
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
        // user's tweaks. peak_pick_delta is skipped until touched.
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

      // 2. Audio context at 48 kHz. Refuse to stream if the browser
      //    silently fell back to a different rate (training mismatch).
      audioCtx = new AudioContext({ sampleRate: REQUIRED_SAMPLE_RATE });
      if (audioCtx.sampleRate !== REQUIRED_SAMPLE_RATE) {
        const got = audioCtx.sampleRate;
        setStatus(
          `error: AudioContext sample rate is ${got} Hz; need ${REQUIRED_SAMPLE_RATE} Hz`,
          "error"
        );
        await stop();
        return;
      }
      await ensureWorklet(audioCtx);

      // 3. Microphone. Request mono + no processing.
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: REQUIRED_SAMPLE_RATE,
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
      // Connect through a zero-gain node so the worklet has a valid
      // graph path to the destination without audible output.
      silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      workletNode.connect(silentGain);
      silentGain.connect(audioCtx.destination);

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
      try { workletNode.disconnect(); } catch (_) { /* noop */ }
      workletNode = null;
    }
    if (silentGain) {
      try { silentGain.disconnect(); } catch (_) { /* noop */ }
      silentGain = null;
    }
    if (source) {
      try { source.disconnect(); } catch (_) { /* noop */ }
      source = null;
    }
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
    // Clear local canvas and scroll watermark so the next trail seeds
    // a fresh history (server total_columns also resets on reset).
    lastEndColumn = -1;
    onsets = [];
    if (colBuffer) colBuffer.fill(DB_MIN);
    if (imageData) {
      const ctx = cqtEl.getContext("2d");
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    }
    clearOnsetLabels();
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

  // ----------------------------------------------------------------- //
  // Theme: light is the default. Choice is stored so a reload keeps it.
  // ----------------------------------------------------------------- //

  const THEME_KEY = "chord-detection-theme";

  function currentTheme() {
    const attr = document.documentElement.getAttribute("data-theme");
    return attr === "dark" ? "dark" : "light";
  }

  function applyTheme(theme) {
    const next = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem(THEME_KEY, next); } catch (_) { /* private mode */ }
    if (themeToggle) {
      const toDark = next === "light";
      themeToggle.textContent = toDark ? "Dark" : "Light";
      themeToggle.setAttribute(
        "aria-label",
        toDark ? "Switch to dark theme" : "Switch to light theme"
      );
      themeToggle.setAttribute("aria-pressed", next === "dark" ? "true" : "false");
    }
  }

  applyTheme(currentTheme());
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      applyTheme(currentTheme() === "dark" ? "light" : "dark");
    });
  }

  // Initial canvas paint.
  const ctx0 = cqtEl.getContext("2d");
  ctx0.fillStyle = "#000";
  ctx0.fillRect(0, 0, CANVAS_W, CANVAS_H);

  if (typeof ResizeObserver !== "undefined" && cqtEl) {
    new ResizeObserver(() => {
      if (lastEndColumn >= 0) drawOnsetLabels();
    }).observe(cqtEl);
  }

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
        // For skip-initial sliders, value 0 means "clear override".
        if (slider.dataset.skipInitial === "true" && v === "0") {
          socket.send(`set ${slider.dataset.paramKey}=none`);
        } else {
          socket.send(v === "" ? `set ${slider.dataset.paramKey}=none` : `set ${slider.dataset.paramKey}=${v}`);
        }
      }
    });
  });
})();
