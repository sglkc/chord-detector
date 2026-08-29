// app.js -- Real-time chord detection demo (frontend).
//
// Responsibilities:
//   1. Capture the microphone via getUserMedia + AudioWorklet.
//   2. Stream Float32 mono PCM to /ws as binary WebSocket frames.
//   3. Render incoming CQT columns on a scrolling canvas.
//   4. Overlay Superflux onsets as red vertical lines with the
//      classified chord for that segment (rotated if it would clip).
//   5. Display the latest chord label.
//   6. Keep a scrollable session list of every classified segment
//      and play back the captured take after Stop.
//   7. Crop each closed segment from the live CQT buffer and show
//      that slice (not the full trail) on the session canvas.
//   8. Download the take as WAV, or upload a file and run it through
//      the same WebSocket pipeline.
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
  const uploadBtn = $("upload");
  const uploadFileEl = $("upload-file");
  const downloadBtn = $("download");
  const playBtn = $("play");
  const seekEl = $("seek");
  const clockEl = $("clock");
  const sessionMetaEl = $("session-meta");
  const segmentMetaEl = $("segment-meta");
  const segmentCqtEl = $("segment-cqt");
  const segmentPlaceholderEl = $("segment-placeholder");
  const chordListEl = $("chord-list");
  const statusEl = $("status");
  const themeToggle = $("theme-toggle");
  const stageEl = document.querySelector(".stage");
  const stageLiveEl = document.querySelector(".stage-live");
  const stageSplitEl = $("stage-split");
  const sessionCardEl = document.querySelector(".session-card");

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
  let uploadCancel = false;
  let lastServerMsgAt = 0;
  let flushWaiters = [];

  // CQT canvas state. We hold a rolling buffer of canvas-width columns.
  // n_bins is learned from the first server message; default 216.
  let nBins = 216;
  // Pixel dimensions of the canvas.
  const CANVAS_W = cqtEl.width;
  const CANVAS_H = cqtEl.height;
  // CNN / segment window length. A full capture uses this many CQT
  // columns; the session canvas maps that span to its full width so
  // a truncated window stays visibly shorter.
  const FEATURE_FRAMES = 188;
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
  const MAX_RECORD_SECONDS = 180;
  const MAX_RECORD_SAMPLES = REQUIRED_SAMPLE_RATE * MAX_RECORD_SECONDS;

  // Client-side take: the same Float32 chunks sent over the socket,
  // kept so Stop can play them back. Chord rows store onset times in
  // this same clock (samples / 48000).
  let recordedChunks = [];
  let recordedSamples = 0;
  let recordCapped = false;
  let chords = [];
  let chordSeq = 0;
  let pendingCaptures = [];
  let shownSegmentId = null;
  let segmentImageData = null;
  let playCtx = null;
  let playSource = null;
  let playStartCtxTime = 0;
  let playOffset = 0;
  let playing = false;
  let seekRaf = 0;
  let seeking = false;

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
    const fontH = 15;
    ctx.font = `600 ${fontH}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.lineWidth = 1.25;
    ctx.lineJoin = "round";
    ctx.miterLimit = 2;
    ctx.strokeStyle = "#000";
    ctx.fillStyle = "#fff";

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

      ctx.save();

      if (fitsH) {
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.strokeText(text, xPx + pad, 4);
        ctx.fillText(text, xPx + pad, 4);
      } else {
        // 90° clockwise: reads top-to-bottom along the onset line.
        // Sit just to the right of the line, or to the left if the
        // right side would clip the canvas edge.
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
  // Segment spectrogram (crop from the live CQT buffer)
  // ----------------------------------------------------------------- //

  function resolveOnsetColumn(msg) {
    if (typeof msg.onset_column === "number" && Number.isFinite(msg.onset_column)) {
      return Math.round(msg.onset_column);
    }
    if (typeof msg.onset_time === "number" && Number.isFinite(msg.onset_time)) {
      return Math.round(msg.onset_time * REQUIRED_SAMPLE_RATE / 512);
    }
    return null;
  }

  function captureSegmentSlice(onsetCol, sourceFrames) {
    if (lastEndColumn < 0 || colBuffer == null || !sourceFrames) return null;
    const nWant = Math.max(1, Math.round(sourceFrames));
    const startX = Math.round(onsetColumnToX(onsetCol));
    const endX = Math.round(onsetColumnToX(onsetCol + nWant));
    if (endX <= 0 || startX >= CANVAS_W) return null;
    const x0 = Math.max(0, startX);
    const x1 = Math.min(CANVAS_W, endX);
    const nCols = x1 - x0;
    if (nCols <= 0) return null;
    const samples = new Float32Array(nBins * nCols);
    samples.set(colBuffer.subarray(x0 * nBins, x1 * nBins));
    return {
      nBins,
      nCols,
      sourceFrames: nWant,
      samples,
      clippedLeft: startX < 0,
      clippedRight: endX > CANVAS_W,
    };
  }

  function tryCaptureChord(entry) {
    if (!entry || entry.thumb) return !!entry.thumb;
    if (typeof entry.onset_column !== "number" || !entry.source_frames) return false;
    if (lastEndColumn < 0 || colBuffer == null) return false;
    const endCol = entry.onset_column + entry.source_frames;
    if (lastEndColumn < endCol) return false;
    const thumb = captureSegmentSlice(entry.onset_column, entry.source_frames);
    if (!thumb || thumb.nCols <= 0) return false;
    entry.thumb = thumb;
    return true;
  }

  function flushPendingCaptures() {
    if (pendingCaptures.length === 0) return;
    const still = [];
    for (let i = 0; i < pendingCaptures.length; i++) {
      const entry = pendingCaptures[i];
      if (tryCaptureChord(entry)) {
        if (shownSegmentId == null || shownSegmentId === entry.id) {
          showSegmentFor(entry);
        }
      } else {
        still.push(entry);
      }
    }
    pendingCaptures = still;
  }

  function finalizeCaptures() {
    for (let i = 0; i < pendingCaptures.length; i++) {
      const entry = pendingCaptures[i];
      if (entry.thumb) continue;
      const thumb = captureSegmentSlice(entry.onset_column, entry.source_frames);
      if (thumb && thumb.nCols > 0) entry.thumb = thumb;
    }
    pendingCaptures = [];
  }

  function clearSegmentCanvas() {
    shownSegmentId = null;
    if (segmentCqtEl) {
      const ctx = segmentCqtEl.getContext("2d");
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, segmentCqtEl.width, segmentCqtEl.height);
    }
    if (segmentPlaceholderEl) {
      segmentPlaceholderEl.hidden = false;
      segmentPlaceholderEl.textContent = "Captured spectrogram";
    }
  }

  function paintSegmentThumb(entry, playheadS) {
    if (!segmentCqtEl || !entry || !entry.thumb) return;
    if (colormap == null) colormap = buildColormap();
    const thumb = entry.thumb;
    const w = segmentCqtEl.width;
    const h = segmentCqtEl.height;
    if (segmentImageData == null || segmentImageData.width !== w || segmentImageData.height !== h) {
      segmentImageData = new ImageData(w, h);
    }
    const data = segmentImageData.data;
    const intended = thumb.sourceFrames || entry.source_frames || thumb.nCols;
    const pxPerFrame = w / FEATURE_FRAMES;
    const padLeft = thumb.clippedLeft ? Math.max(0, intended - thumb.nCols) : 0;

    for (let x = 0; x < w; x++) {
      const frame = Math.floor(x / pxPerFrame) - padLeft;
      if (frame < 0 || frame >= thumb.nCols) {
        for (let y = 0; y < h; y++) {
          const px = (y * w + x) * 4;
          data[px] = 0;
          data[px + 1] = 0;
          data[px + 2] = 0;
          data[px + 3] = 255;
        }
        continue;
      }
      const colStart = frame * thumb.nBins;
      for (let y = 0; y < h; y++) {
        const binF = (y / Math.max(1, h - 1)) * (thumb.nBins - 1);
        const bin0 = Math.floor(binF);
        const bin1 = Math.min(bin0 + 1, thumb.nBins - 1);
        const f = binF - bin0;
        const v = thumb.samples[colStart + bin0] * (1 - f) + thumb.samples[colStart + bin1] * f;
        const t = (v - DB_MIN) / (DB_MAX - DB_MIN);
        const idx = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
        const px = (y * w + x) * 4;
        data[px] = colormap[idx];
        data[px + 1] = colormap[idx + 1];
        data[px + 2] = colormap[idx + 2];
        data[px + 3] = 255;
      }
    }

    const ctx = segmentCqtEl.getContext("2d");
    ctx.putImageData(segmentImageData, 0, 0);

    if (typeof playheadS === "number" && entry.duration > 0) {
      const local = playheadS - entry.onset;
      if (local >= 0 && local <= entry.duration + 1e-3) {
        const x = Math.round((local / entry.duration) * intended * pxPerFrame);
        if (x >= 0 && x < w) {
          ctx.fillStyle = "#fff";
          ctx.fillRect(x, 0, 1, h);
        }
      }
    }

    shownSegmentId = entry.id;
    if (segmentPlaceholderEl) segmentPlaceholderEl.hidden = true;
  }

  function showSegmentFor(entry, playheadS) {
    if (!entry) {
      clearSegmentCanvas();
      return;
    }
    if (entry.thumb) {
      paintSegmentThumb(entry, playheadS);
      return;
    }
    if (shownSegmentId == null && segmentPlaceholderEl) {
      segmentPlaceholderEl.textContent = "Capturing\u2026";
      segmentPlaceholderEl.hidden = false;
    }
  }

  // ----------------------------------------------------------------- //
  // Session list + recorded-audio playback
  // ----------------------------------------------------------------- //

  function recordedDuration() {
    return recordedSamples / REQUIRED_SAMPLE_RATE;
  }

  function formatStrength(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    return value.toFixed(2);
  }

  function formatClock(seconds) {
    const s = Math.max(0, seconds);
    const m = Math.floor(s / 60);
    const frac = (s - m * 60).toFixed(1);
    const padded = frac.length < 4 ? `0${frac}` : frac;
    return `${m}:${padded}`;
  }

  function appendRecordedPcm(f32) {
    if (!f32 || f32.length === 0 || recordCapped) return;
    if (recordedSamples >= MAX_RECORD_SAMPLES) {
      recordCapped = true;
      updateSessionMeta();
      return;
    }
    let chunk = f32;
    if (recordedSamples + f32.length > MAX_RECORD_SAMPLES) {
      chunk = f32.subarray(0, MAX_RECORD_SAMPLES - recordedSamples);
      recordCapped = true;
    }
    recordedChunks.push(chunk);
    recordedSamples += chunk.length;
    if (!playing && !seeking) {
      playOffset = 0;
      updateClock();
    }
    updateSessionMeta();
  }

  function flattenRecording() {
    const out = new Float32Array(recordedSamples);
    let offset = 0;
    for (let i = 0; i < recordedChunks.length; i++) {
      out.set(recordedChunks[i], offset);
      offset += recordedChunks[i].length;
    }
    return out;
  }

  function currentPlayhead() {
    if (playing && playCtx) {
      return Math.min(
        playOffset + (playCtx.currentTime - playStartCtxTime),
        recordedDuration()
      );
    }
    return playOffset;
  }

  function syncTransportEnabled() {
    const canPlay = !running && recordedSamples > 0;
    if (playBtn) playBtn.disabled = !canPlay;
    if (seekEl) seekEl.disabled = !canPlay;
    if (downloadBtn) downloadBtn.disabled = !canPlay;
    if (uploadBtn) uploadBtn.disabled = running;
    if (startBtn && !running) startBtn.disabled = false;
  }

  function updateSessionMeta() {
    if (!sessionMetaEl) return;
    const n = chords.length;
    const noun = n === 1 ? "chord" : "chords";
    const dur = recordedDuration();
    let text = `${n} ${noun}`;
    if (dur > 0) text += ` · ${formatClock(dur)}`;
    if (running) text += " · rec";
    if (recordCapped) text += " · cap";
    sessionMetaEl.textContent = text;
  }

  function updateClock() {
    const dur = recordedDuration();
    const t = running && !playing ? dur : currentPlayhead();
    if (clockEl) {
      clockEl.textContent = dur > 0 && !running
        ? `${formatClock(t)} / ${formatClock(dur)}`
        : formatClock(running ? dur : t);
    }
    if (seekEl && !seeking) {
      seekEl.max = dur > 0 ? String(dur) : "0";
      seekEl.value = String(Math.min(t, dur));
    }
  }

  function updatePlayButton() {
    if (!playBtn) return;
    playBtn.textContent = playing ? "Pause" : "Play";
  }

  function highlightChordAt(timeS) {
    if (!chordListEl) return;
    let activeId = null;
    let active = null;
    if (chords.length > 0) {
      if (timeS < chords[0].onset) {
        active = chords[0];
      } else {
        active = chords[chords.length - 1];
        for (let i = 0; i < chords.length; i++) {
          const c = chords[i];
          const next = chords[i + 1];
          if (next && timeS + 1e-4 >= c.onset && timeS < next.onset) {
            active = c;
            break;
          }
        }
      }
      activeId = active.id;
    }
    chordListEl.querySelectorAll(".chord-row").forEach((el) => {
      el.classList.toggle("is-active", el.dataset.id === String(activeId));
    });
    if (segmentMetaEl) {
      segmentMetaEl.textContent = active
        ? (active.label || active.display_label)
        : "no segment";
    }
    if (active) showSegmentFor(active, timeS);
    else clearSegmentCanvas();
    return active;
  }

  function showChordInHero(msg) {
    if (!msg || !chordEl) return;
    chordEl.textContent = msg.display_label || msg.label || "--";
    const conf = typeof msg.confidence === "number"
      ? `confidence ${(msg.confidence * 100).toFixed(1)}%`
      : "";
    const strength = formatStrength(msg.strength);
    const str = strength != null ? `strength ${strength}` : "";
    const frames = msg.source_frames;
    const win = msg.truncated
      ? `truncated (${frames} frames)`
      : frames
        ? `full (${frames} frames)`
        : "";
    chordMetaEl.textContent = [conf, str, win].filter(Boolean).join("  |  ");
  }

  function tickPlayhead() {
    if (!playing) return;
    const t = currentPlayhead();
    updateClock();
    const active = highlightChordAt(t);
    if (active) showChordInHero(active);
    if (t >= recordedDuration() - 0.01) {
      stopPlayback(false);
      playOffset = recordedDuration();
      updateClock();
      return;
    }
    seekRaf = requestAnimationFrame(tickPlayhead);
  }

  function stopPlayback() {
    const src = playSource;
    playSource = null;
    if (src) {
      src.onended = null;
      try { src.stop(); } catch (_) { /* already stopped */ }
      try { src.disconnect(); } catch (_) { /* noop */ }
    }
    if (playing && playCtx) {
      playOffset = currentPlayhead();
    }
    playing = false;
    if (seekRaf) {
      cancelAnimationFrame(seekRaf);
      seekRaf = 0;
    }
    updatePlayButton();
    updateClock();
    highlightChordAt(playOffset);
  }

  async function playFrom(offsetSec) {
    stopPlayback();
    const dur = recordedDuration();
    if (dur <= 0 || running) return;
    playOffset = Math.max(0, Math.min(offsetSec, dur));
    if (playOffset >= dur) {
      updateClock();
      return;
    }
    if (!playCtx || playCtx.state === "closed") {
      playCtx = new AudioContext({ sampleRate: REQUIRED_SAMPLE_RATE });
    }
    if (playCtx.state === "suspended") {
      try { await playCtx.resume(); } catch (_) { /* autoplay */ }
    }
    const pcm = flattenRecording();
    const buf = playCtx.createBuffer(1, pcm.length, REQUIRED_SAMPLE_RATE);
    buf.getChannelData(0).set(pcm);
    const src = playCtx.createBufferSource();
    src.buffer = buf;
    src.connect(playCtx.destination);
    src.onended = () => {
      if (playSource !== src) return;
      playSource = null;
      playing = false;
      playOffset = recordedDuration();
      updatePlayButton();
      updateClock();
      highlightChordAt(playOffset);
    };
    playSource = src;
    playStartCtxTime = playCtx.currentTime;
    src.start(0, playOffset);
    playing = true;
    updatePlayButton();
    highlightChordAt(playOffset);
    tickPlayhead();
  }

  function emptyChordList() {
    if (!chordListEl) return;
    chordListEl.innerHTML =
      '<li class="chord-empty">Chords appear here as segments close</li>';
  }

  function appendChord(msg) {
    const onset = typeof msg.onset_time === "number" ? msg.onset_time : 0;
    const entry = {
      id: ++chordSeq,
      label: msg.display_label,
      display_label: msg.display_label,
      confidence: msg.confidence,
      onset,
      onset_time: onset,
      duration: typeof msg.duration === "number" ? msg.duration : 0,
      truncated: !!msg.truncated,
      source_frames: typeof msg.source_frames === "number" ? msg.source_frames : 0,
      onset_column: resolveOnsetColumn(msg),
      strength: typeof msg.strength === "number" ? msg.strength : null,
      thumb: null,
    };
    chords.push(entry);
    if (!tryCaptureChord(entry)) pendingCaptures.push(entry);
    if (!chordListEl) {
      updateSessionMeta();
      return entry;
    }
    const empty = chordListEl.querySelector(".chord-empty");
    if (empty) empty.remove();
    const stick =
      chordListEl.scrollHeight - chordListEl.scrollTop - chordListEl.clientHeight < 48;
    const li = document.createElement("li");
    const row = document.createElement("button");
    row.type = "button";
    row.className = "chord-row";
    row.dataset.id = String(entry.id);
    row.dataset.onset = String(entry.onset);
    const confPct =
      typeof entry.confidence === "number"
        ? `${Math.round(entry.confidence * 100)}%`
        : "—";
    const str = formatStrength(entry.strength);
    row.innerHTML =
      `<span class="t">${entry.onset.toFixed(2)}</span>` +
      `<span class="n">${entry.label}</span>` +
      `<span class="c">${confPct}</span>` +
      `<span class="s">${str != null ? str : "—"}</span>` +
      `<span class="d">${entry.duration.toFixed(1)}s</span>`;
    row.title = [
      entry.label,
      typeof entry.confidence === "number"
        ? `${(entry.confidence * 100).toFixed(1)}%`
        : null,
      str != null ? `strength ${str}` : null,
      entry.truncated ? `truncated ${entry.source_frames} frames` : null,
    ].filter(Boolean).join(" · ");
    li.appendChild(row);
    chordListEl.appendChild(li);
    if (stick) chordListEl.scrollTop = chordListEl.scrollHeight;
    updateSessionMeta();
    highlightChordAt(running ? recordedDuration() : playOffset);
    return entry;
  }

  function clearSession() {
    stopPlayback();
    recordedChunks = [];
    recordedSamples = 0;
    recordCapped = false;
    chords = [];
    chordSeq = 0;
    pendingCaptures = [];
    playOffset = 0;
    clearSegmentCanvas();
    emptyChordList();
    updateSessionMeta();
    updateClock();
    syncTransportEnabled();
    if (chordEl) chordEl.textContent = "--";
    if (chordMetaEl) chordMetaEl.textContent = "awaiting audio\u2026";
    if (segmentMetaEl) segmentMetaEl.textContent = "no segment";
  }

  async function closePlayContext() {
    stopPlayback();
    if (playCtx) {
      try { await playCtx.close(); } catch (_) { /* noop */ }
      playCtx = null;
    }
  }

  // ----------------------------------------------------------------- //
  // WebSocket message handling
  // ----------------------------------------------------------------- //

  function handleServerMessage(msg) {
    lastServerMsgAt = Date.now();
    if (msg.type === "flushed") {
      settleFlushWaiters();
    } else if (msg.type === "ready") {
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
          slider.value = slider.dataset.paramKey === "peak_pick_delta" ? "0.07" : "";
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
      flushPendingCaptures();
      const endLabel =
        typeof msg.end_column === "number" ? ` | col ${msg.end_column}` : "";
      cqtMetaEl.textContent =
        `${msg.n_cols} trail | canvas ${CANVAS_W}px${endLabel} | time ${msg.time_s.toFixed(1)}s`;
    } else if (msg.type === "onset") {
      recordOnset(msg.column);
      drawCanvas();
    } else if (msg.type === "chord") {
      showChordInHero(msg);
      appendChord(msg);

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

  const FRAME_MS = 512000 / REQUIRED_SAMPLE_RATE;

  function formatParamValue(key, raw) {
    if (raw === "" || raw == null) return "—";
    if (key === "min_onset_gap_ms") return `${raw} ms`;
    if (key === "peak_pick_delta") {
      const n = Number(raw);
      return Number.isFinite(n) ? n.toFixed(2) : String(raw);
    }
    if (key === "peak_pick_wait") {
      const frames = Number(raw);
      if (!Number.isFinite(frames)) return String(raw);
      return `${frames} frames · ${Math.round(frames * FRAME_MS)} ms`;
    }
    return String(raw);
  }

  function updateSliderLabel(slider) {
    const out = document.querySelector(
      `[data-param-label-for="${slider.dataset.paramKey}"]`
    );
    if (!out) return;
    out.textContent = formatParamValue(slider.dataset.paramKey, slider.value);
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

  function encodeWavPcm16(f32, sampleRate) {
    const n = f32.length;
    const dataBytes = n * 2;
    const buf = new ArrayBuffer(44 + dataBytes);
    const view = new DataView(buf);
    const writeStr = (off, s) => {
      for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + dataBytes, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, dataBytes, true);
    let o = 44;
    for (let i = 0; i < n; i++) {
      let s = f32[i];
      if (s > 1) s = 1;
      else if (s < -1) s = -1;
      view.setInt16(o, s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff), true);
      o += 2;
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  async function decodeToMono48k(file) {
    const raw = await file.arrayBuffer();
    const probe = new AudioContext();
    let decoded;
    try {
      decoded = await probe.decodeAudioData(raw.slice(0));
    } finally {
      try { await probe.close(); } catch (_) { /* noop */ }
    }
    if (decoded.sampleRate === REQUIRED_SAMPLE_RATE && decoded.numberOfChannels === 1) {
      return decoded.getChannelData(0).slice();
    }
    const frames = Math.max(1, Math.ceil(decoded.duration * REQUIRED_SAMPLE_RATE));
    const offline = new OfflineAudioContext(1, frames, REQUIRED_SAMPLE_RATE);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start(0);
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0).slice();
  }

  function downloadRecording() {
    if (running || recordedSamples <= 0) return;
    const pcm = flattenRecording();
    const blob = encodeWavPcm16(pcm, REQUIRED_SAMPLE_RATE);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `chord-session-${stamp}.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function connectPipeline() {
    return new Promise((resolve, reject) => {
      const wsUrl = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
      socket = new WebSocket(wsUrl);
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        sendAllParams();
        resolve();
      };
      socket.onerror = () => {
        setStatus("socket error", "error");
        reject(new Error("socket error"));
      };
      socket.onclose = () => {
        setStatus("disconnected", null);
        if (running) stop();
      };
      socket.onmessage = (ev) => {
        try {
          handleServerMessage(JSON.parse(ev.data));
        } catch (err) {
          setStatus(`bad message: ${err.message}`, "error");
        }
      };
    });
  }

  function settleFlushWaiters() {
    const waiters = flushWaiters;
    flushWaiters = [];
    for (let i = 0; i < waiters.length; i++) waiters[i]();
  }

  function waitFlushed(timeoutMs) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        flushWaiters = flushWaiters.filter((fn) => fn !== onFlush);
        reject(new Error("timed out waiting for server flush"));
      }, timeoutMs);
      const onFlush = () => {
        clearTimeout(timer);
        resolve();
      };
      flushWaiters.push(onFlush);
    });
  }

  async function sendPcmWithBackpressure(pcm) {
    // Server pending queue is 8 chunks. Send fewer than that, then
    // ``flush`` so ingest finishes before more audio arrives.
    const chunk = 4096;
    const batch = 4;
    let inBatch = 0;
    for (let i = 0; i < pcm.length; i += chunk) {
      if (uploadCancel || !socket || socket.readyState !== WebSocket.OPEN) return;
      const slice = pcm.subarray(i, i + chunk);
      socket.send(slice.slice().buffer);
      inBatch += 1;
      if (inBatch >= batch) {
        socket.send("flush");
        await waitFlushed(30000);
        inBatch = 0;
      }
    }
    if (inBatch > 0 && socket && socket.readyState === WebSocket.OPEN) {
      socket.send("flush");
      await waitFlushed(30000);
    }
  }

  async function processUploadedFile(file) {
    if (running || !file) return;
    uploadCancel = false;
    stopPlayback();
    setStatus("decoding audio...", null);
    let pcm;
    try {
      pcm = await decodeToMono48k(file);
    } catch (err) {
      setStatus(`error: could not read audio (${err.message})`, "error");
      return;
    }
    if (!pcm || pcm.length === 0) {
      setStatus("error: empty audio file", "error");
      return;
    }
    if (pcm.length > MAX_RECORD_SAMPLES) {
      pcm = pcm.subarray(0, MAX_RECORD_SAMPLES);
    }

    running = true;
    resetBtn.disabled = false;
    if (startBtn) startBtn.disabled = true;
    syncTransportEnabled();
    lastEndColumn = -1;
    onsets = [];
    if (colBuffer) colBuffer.fill(DB_MIN);
    clearOnsetLabels();
    await closePlayContext();
    clearSession();
    recordedChunks = [pcm];
    recordedSamples = pcm.length;
    updateSessionMeta();
    updateClock();

    try {
      setStatus("processing file...", "connected");
      await connectPipeline();
      if (uploadCancel) {
        await stop();
        return;
      }
      await sendPcmWithBackpressure(pcm);
      // Last CQT / chord frames can arrive just after flush.
      await sleep(400);
    } catch (err) {
      setStatus(`error: ${err.message}`, "error");
      console.error(err);
    }
    await stop();
    if (!uploadCancel) setStatus("file processed", "connected");
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
    stopPlayback();
    running = true;
    setStartButtonState(true);
    resetBtn.disabled = false;
    syncTransportEnabled();
    updateSessionMeta();
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

      // New take only after the mic is actually granted, so a
      // dismissed permission prompt does not wipe a reviewable session.
      await closePlayContext();
      clearSession();

      const src = audioCtx.createMediaStreamSource(mediaStream);
      source = src;
      workletNode = new AudioWorkletNode(audioCtx, "mic-processor", {
        processorOptions: { chunkSize: 4096 },
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      workletNode.port.onmessage = (ev) => {
        const f32 = ev.data;
        // Same buffer the worklet transferred to us. Keep a reference
        // for later playback, then ship it to the server (send copies).
        appendRecordedPcm(f32);
        if (!socket || socket.readyState !== WebSocket.OPEN) return;
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
    uploadCancel = true;
    settleFlushWaiters();
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
    playOffset = 0;
    finalizeCaptures();
    syncTransportEnabled();
    updateClock();
    updateSessionMeta();
    if (chords.length > 0) {
      const last = chords[chords.length - 1];
      showSegmentFor(last, last.onset);
    }
  }

  function reset() {
    uploadCancel = true;
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
    clearSession();
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

  if (downloadBtn) {
    downloadBtn.addEventListener("click", downloadRecording);
  }

  if (uploadBtn && uploadFileEl) {
    uploadBtn.addEventListener("click", () => {
      if (running) return;
      uploadFileEl.value = "";
      uploadFileEl.click();
    });
    uploadFileEl.addEventListener("change", () => {
      const file = uploadFileEl.files && uploadFileEl.files[0];
      if (file) processUploadedFile(file);
    });
  }

  if (playBtn) {
    playBtn.addEventListener("click", () => {
      if (running || recordedSamples <= 0) return;
      if (playing) {
        stopPlayback();
        return;
      }
      const dur = recordedDuration();
      const from = playOffset >= dur - 0.02 ? 0 : playOffset;
      playFrom(from);
    });
  }

  if (seekEl) {
    seekEl.addEventListener("input", () => {
      seeking = true;
      playOffset = parseFloat(seekEl.value) || 0;
      updateClock();
      highlightChordAt(playOffset);
    });
    seekEl.addEventListener("change", () => {
      seeking = false;
      const t = parseFloat(seekEl.value) || 0;
      if (playing) playFrom(t);
      else {
        playOffset = t;
        updateClock();
        highlightChordAt(t);
      }
    });
  }

  if (chordListEl) {
    chordListEl.addEventListener("click", (ev) => {
      const row = ev.target.closest(".chord-row");
      if (!row || running) return;
      const t = parseFloat(row.dataset.onset);
      if (!Number.isFinite(t)) return;
      playFrom(t);
    });
  }

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
  if (segmentCqtEl) {
    const sctx = segmentCqtEl.getContext("2d");
    sctx.fillStyle = "#000";
    sctx.fillRect(0, 0, segmentCqtEl.width, segmentCqtEl.height);
  }

  if (typeof ResizeObserver !== "undefined" && cqtEl) {
    new ResizeObserver(() => {
      if (lastEndColumn >= 0) drawOnsetLabels();
    }).observe(cqtEl);
  }

  const SPLIT_KEY = "chord-detection-split";
  const SPLIT_MIN = 0.28;
  const SPLIT_MAX = 0.72;
  const SPLIT_DEFAULT = 0.58;
  const stackedMq = window.matchMedia("(max-width: 880px)");

  function readSplitFrac() {
    try {
      const raw = localStorage.getItem(SPLIT_KEY);
      const n = raw == null ? SPLIT_DEFAULT : parseFloat(raw);
      if (Number.isFinite(n)) return Math.max(SPLIT_MIN, Math.min(SPLIT_MAX, n));
    } catch (_) { /* private mode */ }
    return SPLIT_DEFAULT;
  }

  function writeSplitFrac(frac) {
    try { localStorage.setItem(SPLIT_KEY, String(frac)); } catch (_) { /* private mode */ }
  }

  function applySplit(frac) {
    if (!stageEl || !stageLiveEl || !sessionCardEl || !stageSplitEl) return;
    if (stackedMq.matches) {
      stageLiveEl.style.flex = "";
      sessionCardEl.style.flex = "";
      return;
    }
    const gutter = stageSplitEl.offsetWidth || 28;
    const total = stageEl.clientWidth - gutter;
    if (total <= 0) return;
    const clamped = Math.max(SPLIT_MIN, Math.min(SPLIT_MAX, frac));
    const livePx = Math.round(clamped * total);
    stageLiveEl.style.flex = `0 0 ${livePx}px`;
    sessionCardEl.style.flex = `1 1 auto`;
    if (stageSplitEl) {
      stageSplitEl.setAttribute("aria-valuenow", String(Math.round(clamped * 100)));
    }
  }

  function bindStageSplit() {
    if (!stageEl || !stageSplitEl) return;
    stageSplitEl.setAttribute("aria-valuemin", String(Math.round(SPLIT_MIN * 100)));
    stageSplitEl.setAttribute("aria-valuemax", String(Math.round(SPLIT_MAX * 100)));
    applySplit(readSplitFrac());

    let dragging = false;

    const onMove = (clientX) => {
      const rect = stageEl.getBoundingClientRect();
      const gutter = stageSplitEl.offsetWidth || 28;
      const total = rect.width - gutter;
      if (total <= 0) return;
      const frac = (clientX - rect.left) / total;
      const clamped = Math.max(SPLIT_MIN, Math.min(SPLIT_MAX, frac));
      applySplit(clamped);
      writeSplitFrac(clamped);
    };

    stageSplitEl.addEventListener("pointerdown", (ev) => {
      if (stackedMq.matches) return;
      dragging = true;
      stageEl.classList.add("is-resizing");
      try { stageSplitEl.setPointerCapture(ev.pointerId); } catch (_) { /* noop */ }
      ev.preventDefault();
    });
    stageSplitEl.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      onMove(ev.clientX);
    });
    const endDrag = (ev) => {
      if (!dragging) return;
      dragging = false;
      stageEl.classList.remove("is-resizing");
      if (ev && ev.pointerId != null) {
        try { stageSplitEl.releasePointerCapture(ev.pointerId); } catch (_) { /* noop */ }
      }
    };
    stageSplitEl.addEventListener("pointerup", endDrag);
    stageSplitEl.addEventListener("pointercancel", endDrag);
    stageSplitEl.addEventListener("dblclick", () => {
      applySplit(SPLIT_DEFAULT);
      writeSplitFrac(SPLIT_DEFAULT);
    });
    stageSplitEl.addEventListener("keydown", (ev) => {
      if (stackedMq.matches) return;
      const step = ev.shiftKey ? 0.08 : 0.03;
      let frac = readSplitFrac();
      if (ev.key === "ArrowLeft") frac -= step;
      else if (ev.key === "ArrowRight") frac += step;
      else if (ev.key === "Home") frac = SPLIT_MIN;
      else if (ev.key === "End") frac = SPLIT_MAX;
      else if (ev.key === "Enter" || ev.key === " ") frac = SPLIT_DEFAULT;
      else return;
      ev.preventDefault();
      applySplit(frac);
      writeSplitFrac(frac);
    });

    window.addEventListener("resize", () => applySplit(readSplitFrac()));
    if (typeof stackedMq.addEventListener === "function") {
      stackedMq.addEventListener("change", () => applySplit(readSplitFrac()));
    } else if (typeof stackedMq.addListener === "function") {
      stackedMq.addListener(() => applySplit(readSplitFrac()));
    }
  }
  bindStageSplit();

  window.addEventListener("demo:session", (ev) => {
    const d = (ev && ev.detail) || {};
    if (running) return;
    if (typeof d.duration === "number" && d.duration > 0) {
      const n = Math.max(1, Math.floor(d.duration * REQUIRED_SAMPLE_RATE));
      appendRecordedPcm(new Float32Array(n));
      syncTransportEnabled();
      updateClock();
    }
    if (Array.isArray(d.chords)) {
      d.chords.forEach((c) => {
        handleServerMessage({
          type: "chord",
          display_label: c.display_label || c.label,
          confidence: c.confidence,
          onset_time: c.onset_time != null ? c.onset_time : c.onset,
          duration: c.duration,
          truncated: !!c.truncated,
          source_frames: c.source_frames || 188,
          onset_column: c.onset_column,
          strength: c.strength,
        });
      });
      chords.forEach((entry) => {
        if (entry.thumb) return;
        const n = Math.max(1, entry.source_frames || FEATURE_FRAMES);
        const samples = new Float32Array(nBins * n);
        samples.fill(DB_MIN);
        const ridges = [36, 72, 108, 144];
        for (let col = 0; col < n; col++) {
          const env = Math.exp(-2.4 * col / n);
          for (let r = 0; r < ridges.length; r++) {
            for (let d = -3; d <= 3; d++) {
              const b = ridges[r] + d;
              if (b < 0 || b >= nBins) continue;
              const val = DB_MIN + (0 - DB_MIN) * env * (1 - Math.abs(d) / 4);
              const idx = col * nBins + (nBins - 1 - b);
              if (val > samples[idx]) samples[idx] = val;
            }
          }
        }
        entry.thumb = {
          nBins,
          nCols: n,
          sourceFrames: n,
          samples,
          clippedLeft: false,
          clippedRight: false,
        };
      });
      pendingCaptures = pendingCaptures.filter((e) => !e.thumb);
      highlightChordAt(running ? recordedDuration() : playOffset);
    }
  });

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
