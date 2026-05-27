/**
 * Bulk Validation App
 * Validate multiple audio-annotation pairs at once.
 * Matches WAV+TXT by filename, processes async with progressive card rendering.
 * 
 * Config is shared across pages via sessionStorage.
 */

import { WCSRValidator } from './modules/wcsr-validator.js';
import { Visualizer } from './modules/visualizer.js';
import { CONFIG } from './modules/config.js';
import { ClassificationService } from './modules/classification-service.js';

const SESSION_KEY = 'chordDetectionConfig';

class BulkValidationApp {
  constructor() {
    this.rawFiles = [];           // All uploaded File objects
    this.pairs = [];              // { baseName, wavFile, txtFile }
    this.unmatched = [];          // Files with no pair
    this.results = new Map();     // cardId → { wcsr, results, predictions, ... }
    this.audioBuffers = new Map();

    // Playback state
    this.audioContext = null;
    this.currentlyPlaying = null;
    this.currentSource = null;
    this.currentAnimFrame = null;

    // Running stats
    this.processedCount = 0;
    this.wcsrValues = [];

    // Progress state — tracks per-file sub-progress from worker
    this.currentFileIndex = 0;
    this.totalFiles = 0;

    // Modules
    this.validator = new WCSRValidator();
    this.visualizer = new Visualizer();
    this.classificationService = new ClassificationService({
      model: CONFIG.classification.model,
      onProgress: (percent, message, stage) => {
        this.handleWorkerProgress(percent, message, stage);
      }
    });

    this.initElements();
    this.loadConfigFromSession();
    this.bindEvents();
  }

  // ── DOM refs ─────────────────────────────────────────────
  initElements() {
    this.uploadZone = document.getElementById('uploadZone');
    this.fileInput = document.getElementById('bulkFiles');
    this.fileCount = document.getElementById('fileCount');
    this.runBtn = document.getElementById('runBtn');
    this.clearBtn = document.getElementById('clearBtn');

    this.pairingSummary = document.getElementById('pairingSummary');
    this.pairingList = document.getElementById('pairingList');
    this.matchedCountEl = document.getElementById('matchedCount');
    this.unmatchedCountEl = document.getElementById('unmatchedCount');

    this.progressSection = document.getElementById('progressSection');
    this.progressFill = document.getElementById('progressFill');
    this.progressText = document.getElementById('progressText');
    this.progressPercent = document.getElementById('progressPercent');

    this.summaryBar = document.getElementById('summaryBar');
    this.summaryProcessed = document.getElementById('summaryProcessed');
    this.summaryAvgWcsr = document.getElementById('summaryAvgWcsr');
    this.summaryBest = document.getElementById('summaryBest');
    this.summaryWorst = document.getElementById('summaryWorst');

    this.resultsSection = document.getElementById('resultsSection');
    this.resultsGrid = document.getElementById('resultsGrid');

    // Config inputs
    this.configInputs = {
      onsetThreshold: document.getElementById('onsetThreshold'),
      ignoreSubsequentOnsets: document.getElementById('ignoreSubsequentOnsets'),
      flexibleWindow: document.getElementById('flexibleWindow'),
    };
  }

  // ── Config / SessionStorage ──────────────────────────────
  loadConfigFromSession() {
    try {
      const saved = sessionStorage.getItem(SESSION_KEY);
      if (!saved) return;
      const data = JSON.parse(saved);

      // Apply to CONFIG
      if (data.onset) Object.assign(CONFIG.onset, data.onset);
      if (data.classification) Object.assign(CONFIG.classification, data.classification);

      // Apply to UI
      this.configInputs.onsetThreshold.value = CONFIG.onset.threshold;
      this.configInputs.ignoreSubsequentOnsets.checked = CONFIG.onset.ignoreSubsequentOnsets;
      this.configInputs.flexibleWindow.checked = CONFIG.classification.flexibleWindow || false;
      this.configInputs.ignoreSubsequentOnsets.disabled = CONFIG.classification.flexibleWindow || false;
    } catch (e) {
      console.warn('Failed to load config from session:', e);
    }
  }

  /** Read UI inputs → CONFIG object → sessionStorage */
  updateConfig() {
    CONFIG.onset.threshold = parseFloat(this.configInputs.onsetThreshold.value);
    CONFIG.onset.ignoreSubsequentOnsets = this.configInputs.ignoreSubsequentOnsets.checked;
    CONFIG.classification.flexibleWindow = this.configInputs.flexibleWindow.checked;

    // Mutual exclusion: flexible window disables ignore subsequent onsets
    this.configInputs.ignoreSubsequentOnsets.disabled = CONFIG.classification.flexibleWindow;

    this.saveConfigToSession();
  }

  saveConfigToSession() {
    try {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify({
        onset: {
          threshold: CONFIG.onset.threshold,
          minInterval: CONFIG.onset.minInterval,
          preBuffer: CONFIG.onset.preBuffer,
          ignoreSubsequentOnsets: CONFIG.onset.ignoreSubsequentOnsets,
        },
        classification: {
          windowSize: CONFIG.classification.windowSize,
          flexibleWindow: CONFIG.classification.flexibleWindow,
        },
      }));
    } catch (e) {
      console.warn('Failed to save config to session:', e);
    }
  }

  // ── Events ───────────────────────────────────────────────
  bindEvents() {
    this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

    this.uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.uploadZone.classList.add('drag-over');
    });
    this.uploadZone.addEventListener('dragleave', () => {
      this.uploadZone.classList.remove('drag-over');
    });
    this.uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.uploadZone.classList.remove('drag-over');
      if (e.dataTransfer.files.length > 0) {
        this.addFiles(Array.from(e.dataTransfer.files));
      }
    });

    this.runBtn.addEventListener('click', () => this.runAll());
    this.clearBtn.addEventListener('click', () => this.clearAll());
  }

  // ── File handling ────────────────────────────────────────
  handleFileSelect(event) {
    this.addFiles(Array.from(event.target.files));
  }

  addFiles(files) {
    // Filter to wav + txt only
    const valid = files.filter(f => {
      const ext = f.name.split('.').pop().toLowerCase();
      return ext === 'wav' || ext === 'ogg' || ext === 'txt';
    });

    if (valid.length === 0) {
      alert('Please select WAV and/or TXT files.');
      return;
    }

    this.rawFiles = [...this.rawFiles, ...valid];
    this.buildPairs();
    this.renderPairingSummary();
    this.updateFileCount();

    this.runBtn.disabled = this.pairs.length === 0;
    this.clearBtn.disabled = false;
  }

  /**
   * Group uploaded files by base filename (without extension).
   * A valid pair has both a .wav and a .txt with the same base name.
   */
  buildPairs() {
    const map = new Map(); // baseName → { wav, txt }

    for (const file of this.rawFiles) {
      const dotIdx = file.name.lastIndexOf('.');
      const baseName = dotIdx > 0 ? file.name.substring(0, dotIdx) : file.name;
      const ext = dotIdx > 0 ? file.name.substring(dotIdx + 1).toLowerCase() : '';

      if (!map.has(baseName)) map.set(baseName, {});
      const entry = map.get(baseName);

      if (ext === 'wav') entry.wav = file;
      else if (ext === 'txt') entry.txt = file;
    }

    this.pairs = [];
    this.unmatched = [];

    for (const [baseName, entry] of map) {
      if (entry.wav && entry.txt) {
        this.pairs.push({ baseName, wavFile: entry.wav, txtFile: entry.txt });
      } else {
        if (entry.wav) this.unmatched.push({ file: entry.wav, type: 'wav', baseName });
        if (entry.txt) this.unmatched.push({ file: entry.txt, type: 'txt', baseName });
      }
    }

    // Sort pairs alphabetically
    this.pairs.sort((a, b) => a.baseName.localeCompare(b.baseName));
  }

  renderPairingSummary() {
    this.pairingSummary.style.display = 'block';
    this.matchedCountEl.textContent = `${this.pairs.length} matched pair${this.pairs.length !== 1 ? 's' : ''}`;

    if (this.unmatched.length > 0) {
      this.unmatchedCountEl.style.display = '';
      this.unmatchedCountEl.textContent = `${this.unmatched.length} unmatched`;
    } else {
      this.unmatchedCountEl.style.display = 'none';
    }

    this.pairingList.innerHTML = '';

    for (const pair of this.pairs) {
      const el = document.createElement('div');
      el.className = 'pairing-item matched';
      el.innerHTML = `
        <span class="pairing-icon">✅</span>
        <span class="pairing-name" title="${this.esc(pair.baseName)}">${this.esc(pair.baseName)}</span>
        <span class="pairing-status">.wav + .txt</span>
      `;
      this.pairingList.appendChild(el);
    }

    for (const item of this.unmatched) {
      const el = document.createElement('div');
      el.className = 'pairing-item unmatched';
      el.innerHTML = `
        <span class="pairing-icon">⚠️</span>
        <span class="pairing-name" title="${this.esc(item.file.name)}">${this.esc(item.file.name)}</span>
        <span class="pairing-status">missing .${item.type === 'wav' ? 'txt' : 'wav'}</span>
      `;
      this.pairingList.appendChild(el);
    }
  }

  updateFileCount() {
    this.fileCount.textContent = `${this.rawFiles.length} file${this.rawFiles.length !== 1 ? 's' : ''} selected`;
  }

  clearAll() {
    this.rawFiles = [];
    this.pairs = [];
    this.unmatched = [];
    this.results.clear();
    this.audioBuffers.clear();
    this.processedCount = 0;
    this.wcsrValues = [];
    this.stopCurrentAudio();

    this.fileInput.value = '';
    this.fileCount.textContent = '0 files selected';
    this.runBtn.disabled = true;
    this.clearBtn.disabled = true;
    this.pairingSummary.style.display = 'none';
    this.progressSection.style.display = 'none';
    this.summaryBar.style.display = 'none';
    this.resultsSection.style.display = 'none';
    this.resultsGrid.innerHTML = '';
  }

  // ── Progress (with worker sub-progress forwarding) ───────
  /**
   * Maps the worker's per-file progress (0–100) into a slice of the
   * overall progress bar for that file.
   *
   * Layout:  [0–5% init] [5–95% files] [95–100% done]
   * Per file band = 90% / totalFiles
   */
  handleWorkerProgress(workerPercent, message, stage) {
    if (this.totalFiles === 0) return;

    // During init stage, show directly
    if (stage === 'init') {
      this.updateProgress(workerPercent * 0.05, message);
      return;
    }

    const bandSize = 90 / this.totalFiles;               // % per file
    const fileBase = 5 + this.currentFileIndex * bandSize; // start of this file's band
    const overall = fileBase + (workerPercent / 100) * bandSize;

    this.updateProgress(overall, `[${this.pairs[this.currentFileIndex]?.baseName}] ${message}`);
  }

  // ── Main processing loop ─────────────────────────────────
  async runAll() {
    if (this.pairs.length === 0) return;

    // Read UI → CONFIG → sessionStorage
    this.updateConfig();

    this.runBtn.disabled = true;
    this.clearBtn.disabled = true;
    this.processedCount = 0;
    this.wcsrValues = [];
    this.results.clear();
    this.audioBuffers.clear();
    this.resultsGrid.innerHTML = '';

    this.progressSection.style.display = 'block';
    this.summaryBar.style.display = 'flex';
    this.resultsSection.style.display = 'block';

    this.totalFiles = this.pairs.length;

    try {
      // Init service
      this.updateProgress(1, 'Loading model...');
      await this.initService();

      for (let i = 0; i < this.totalFiles; i++) {
        const pair = this.pairs[i];
        const cardId = `card-${i}`;
        this.currentFileIndex = i;

        // Create placeholder card
        this.createCardPlaceholder(cardId, pair.baseName);

        try {
          // Load audio
          const audioBuffer = await this.loadAudio(pair.wavFile);
          this.audioBuffers.set(cardId, audioBuffer);

          // Parse annotations
          const annotationText = await pair.txtFile.text();
          const annotations = this.parseAnnotations(annotationText);

          // Process audio (CQT + onset detection + classification)
          // Worker progress is forwarded via handleWorkerProgress
          this.classificationService.updateConfig(CONFIG);
          const audioData = audioBuffer.getChannelData(0);
          const { fullCQT, onsets, predictions } = await this.classificationService.processAudio(
            audioData,
            audioBuffer.duration
          );

          // Calculate WCSR
          const wcsrResults = this.validator.calculate(predictions, annotations);

          // Store
          this.results.set(cardId, {
            wcsrResults,
            predictions,
            annotations,
            fullCQT,
            onsets,
            duration: audioBuffer.duration
          });

          // Update card
          this.updateCard(cardId, pair.baseName, audioBuffer.duration, wcsrResults);

          // Update running stats
          this.processedCount++;
          this.wcsrValues.push(wcsrResults.wcsr);
          this.updateSummary(this.totalFiles);

        } catch (error) {
          console.error(`Error processing ${pair.baseName}:`, error);
          this.updateCardError(cardId, error.message);
          this.processedCount++;
          this.updateSummary(this.totalFiles);
        }

        // Yield to UI
        await new Promise(r => setTimeout(r, 0));
      }

      this.updateProgress(100, 'All validations complete!');
      setTimeout(() => { this.progressSection.style.display = 'none'; }, 1500);

    } catch (error) {
      console.error('Bulk validation error:', error);
      this.updateProgress(0, `Error: ${error.message}`);
    } finally {
      this.runBtn.disabled = false;
      this.clearBtn.disabled = false;
    }
  }

  async initService() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    await this.classificationService.init();
  }

  async loadAudio(file) {
    const arrayBuffer = await file.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: CONFIG.audio.sampleRate
    });
    return await ctx.decodeAudioData(arrayBuffer);
  }

  parseAnnotations(text) {
    const lines = text.trim().split('\n');
    const annotations = [];
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 3) {
        const start = parseFloat(parts[0]);
        const end = parseFloat(parts[1]);
        const chord = parts[2];
        if (!isNaN(start) && !isNaN(end)) {
          annotations.push({ start, end, chord });
        }
      }
    }
    return annotations;
  }

  // ── Card rendering ───────────────────────────────────────
  createCardPlaceholder(cardId, baseName) {
    const card = document.createElement('div');
    card.id = cardId;
    card.className = 'validation-card loading';
    card.innerHTML = `
      <div class="card-collapsed">
        <span class="card-expand-icon">▶</span>
        <span class="card-filename" title="${this.esc(baseName)}">${this.esc(baseName)}</span>
        <span class="card-duration">--</span>
        <span class="card-wcsr-badge pending">...</span>
        <span class="card-status-icon"><span class="card-spinner"></span></span>
      </div>
      <div class="card-detail"></div>
    `;
    this.resultsGrid.appendChild(card);
  }

  updateCard(cardId, baseName, duration, wcsrResults) {
    const card = document.getElementById(cardId);
    if (!card) return;

    card.className = 'validation-card';
    const wcsr = wcsrResults.wcsr * 100;
    const wcsrClass = wcsr >= 80 ? 'high' : wcsr >= 60 ? 'medium' : 'low';

    // Update collapsed row
    const collapsed = card.querySelector('.card-collapsed');
    collapsed.innerHTML = `
      <span class="card-expand-icon">▶</span>
      <span class="card-filename" title="${this.esc(baseName)}">${this.esc(baseName)}</span>
      <span class="card-duration">${duration.toFixed(1)}s</span>
      <span class="card-wcsr-badge ${wcsrClass}">${wcsr.toFixed(1)}%</span>
      <span class="card-status-icon">✅</span>
    `;

    // Build detail pane (rendered lazily on first expand)
    const detail = card.querySelector('.card-detail');
    detail.innerHTML = `
      <!-- Score row -->
      <div class="card-score-row">
        <div class="card-score-item">
          <span class="card-score-label">WCSR</span>
          <span class="card-score-value">${wcsr.toFixed(1)}%</span>
        </div>
        <div class="card-score-item">
          <span class="card-score-label">Total Duration</span>
          <span class="card-score-value">${wcsrResults.totalDuration.toFixed(1)}s</span>
        </div>
        <div class="card-score-item">
          <span class="card-score-label">Correct Duration</span>
          <span class="card-score-value">${wcsrResults.correctDuration.toFixed(1)}s</span>
        </div>
        <div class="card-score-item">
          <span class="card-score-label">Chords</span>
          <span class="card-score-value">${wcsrResults.numPredictions}</span>
        </div>
      </div>

      <!-- CQT -->
      <div class="card-cqt-section">
        <div class="card-cqt-controls">
          <button class="card-play-btn" data-card-id="${cardId}" title="Play audio">
            <span class="play-icon">▶</span>
          </button>
          <span class="card-playback-time">0:00.000 / ${this.formatTime(duration)}</span>
        </div>
        <div class="card-cqt-wrapper">
          <canvas class="card-cqt-canvas" id="canvas-${cardId}"></canvas>
          <div class="card-playback-cursor" id="cursor-${cardId}"></div>
        </div>
        <div class="card-cqt-legend">
          <span class="card-legend-item"><span class="card-legend-color onset"></span> Onsets</span>
          <span class="card-legend-item"><span class="card-legend-color gt"></span> Ground Truth</span>
        </div>
      </div>

      <!-- Timeline -->
      <div class="card-timeline-section">
        <h4>🎼 Chord Timeline</h4>
        <div class="card-timeline-container">
          <div class="card-timeline-row">
            <span class="card-timeline-label">Ground Truth</span>
            <div class="card-timeline-track" id="gt-${cardId}"></div>
          </div>
          <div class="card-timeline-row">
            <span class="card-timeline-label">Predicted</span>
            <div class="card-timeline-track" id="pred-${cardId}"></div>
          </div>
          <div class="card-timeline-axis" id="axis-${cardId}"></div>
        </div>
      </div>

      <!-- Per-chord accuracy -->
      <div class="card-accuracy-section">
        <h4>🎯 Per-Chord Accuracy</h4>
        <div class="card-accuracy-grid" id="accuracy-${cardId}"></div>
      </div>
    `;

    // Bind expand/collapse
    collapsed.addEventListener('click', () => this.toggleCard(cardId));

    // Track whether detail was rendered
    card.dataset.detailRendered = 'false';
  }

  updateCardError(cardId, errorMessage) {
    const card = document.getElementById(cardId);
    if (!card) return;

    card.className = 'validation-card error';
    const collapsed = card.querySelector('.card-collapsed');
    const statusIcon = collapsed.querySelector('.card-status-icon');
    const badge = collapsed.querySelector('.card-wcsr-badge');
    const spinner = collapsed.querySelector('.card-spinner');

    if (statusIcon) statusIcon.innerHTML = '❌';
    if (badge) { badge.className = 'card-wcsr-badge low'; badge.textContent = 'ERR'; }
    if (spinner) spinner.remove();

    const detail = card.querySelector('.card-detail');
    detail.innerHTML = `<div class="card-error-message">⚠️ ${this.esc(errorMessage)}</div>`;

    // Bind expand even on error
    collapsed.addEventListener('click', () => this.toggleCard(cardId));
  }

  // ── Expand / Collapse ────────────────────────────────────
  toggleCard(cardId) {
    const card = document.getElementById(cardId);
    if (!card) return;

    const isExpanded = card.classList.contains('expanded');

    if (!isExpanded) {
      card.classList.add('expanded');
      // Lazy-render heavy visualizations on first expand
      if (card.dataset.detailRendered === 'false' && this.results.has(cardId)) {
        this.renderCardDetail(cardId);
        card.dataset.detailRendered = 'true';
      }
    } else {
      card.classList.remove('expanded');
    }
  }

  renderCardDetail(cardId) {
    const data = this.results.get(cardId);
    if (!data) return;

    const { fullCQT, onsets, annotations, predictions, wcsrResults, duration } = data;

    // Draw CQT
    const canvas = document.getElementById(`canvas-${cardId}`);
    if (canvas) {
      // Set actual pixel dimensions
      canvas.width = canvas.offsetWidth || 1000;
      canvas.height = 200;
      this.visualizer.drawCQT(canvas, fullCQT, onsets, annotations, duration);
    }

    // Draw timeline
    this.visualizer.drawTimeline(
      document.getElementById(`gt-${cardId}`),
      document.getElementById(`pred-${cardId}`),
      document.getElementById(`axis-${cardId}`),
      annotations,
      predictions,
      duration
    );

    // Draw per-chord accuracy
    this.drawAccuracyGrid(
      document.getElementById(`accuracy-${cardId}`),
      wcsrResults.perChordStats
    );

    // Bind play button
    const playBtn = document.querySelector(`#${cardId} .card-play-btn`);
    if (playBtn) {
      playBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.togglePlay(cardId);
      });
    }

    // Bind canvas click for seeking
    if (canvas) {
      canvas.addEventListener('click', (e) => {
        e.stopPropagation();
        this.handleCanvasClick(cardId, e);
      });
    }
  }

  drawAccuracyGrid(container, perChordStats) {
    if (!container) return;
    container.innerHTML = '';

    const sorted = Object.entries(perChordStats)
      .sort((a, b) => b[1].accuracy - a[1].accuracy);

    for (const [chord, stats] of sorted) {
      const accuracy = stats.accuracy * 100;
      const color = accuracy >= 80 ? '#27ae60' : accuracy >= 60 ? '#f39c12' : accuracy >= 40 ? '#e67e22' : '#e74c3c';

      const item = document.createElement('div');
      item.className = 'card-accuracy-item';
      item.innerHTML = `
        <div class="chord-name">${chord}</div>
        <div class="chord-stats">${stats.correctCount}/${stats.count} (${accuracy.toFixed(0)}%)</div>
        <div class="accuracy-bar">
          <div class="accuracy-fill" style="width: ${accuracy}%; background: ${color}"></div>
        </div>
      `;
      container.appendChild(item);
    }
  }

  // ── Audio playback ───────────────────────────────────────
  togglePlay(cardId) {
    if (this.currentlyPlaying === cardId) {
      this.stopCurrentAudio();
    } else {
      this.playAudio(cardId);
    }
  }

  playAudio(cardId) {
    this.stopCurrentAudio();

    const audioBuffer = this.audioBuffers.get(cardId);
    if (!audioBuffer) return;

    if (!this.audioContext || this.audioContext.state === 'closed') {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }

    this.currentSource = this.audioContext.createBufferSource();
    this.currentSource.buffer = audioBuffer;
    this.currentSource.connect(this.audioContext.destination);

    const playBtn = document.querySelector(`#${cardId} .card-play-btn`);
    if (playBtn) {
      playBtn.classList.add('playing');
      playBtn.querySelector('.play-icon').textContent = '⏹';
    }

    const cursor = document.getElementById(`cursor-${cardId}`);
    if (cursor) cursor.classList.add('visible');

    this.currentlyPlaying = cardId;
    this.playbackStartTime = this.audioContext.currentTime;

    this.currentSource.onended = () => {
      if (this.currentlyPlaying === cardId) {
        this.stopCurrentAudio();
      }
    };

    this.currentSource.start(0);
    this.updatePlaybackCursor(cardId);
  }

  stopCurrentAudio() {
    if (this.currentSource) {
      try { this.currentSource.stop(); } catch (e) { /* ok */ }
      this.currentSource.disconnect();
      this.currentSource = null;
    }

    if (this.currentAnimFrame) {
      cancelAnimationFrame(this.currentAnimFrame);
      this.currentAnimFrame = null;
    }

    if (this.currentlyPlaying) {
      const playBtn = document.querySelector(`#${this.currentlyPlaying} .card-play-btn`);
      if (playBtn) {
        playBtn.classList.remove('playing');
        playBtn.querySelector('.play-icon').textContent = '▶';
      }
      const cursor = document.getElementById(`cursor-${this.currentlyPlaying}`);
      if (cursor) cursor.classList.remove('visible');

      this.currentlyPlaying = null;
    }
  }

  updatePlaybackCursor(cardId) {
    if (this.currentlyPlaying !== cardId) return;

    const audioBuffer = this.audioBuffers.get(cardId);
    if (!audioBuffer) return;

    const elapsed = this.audioContext.currentTime - this.playbackStartTime;
    const duration = audioBuffer.duration;

    if (elapsed >= duration) {
      this.stopCurrentAudio();
      return;
    }

    const pct = (elapsed / duration) * 100;
    const cursor = document.getElementById(`cursor-${cardId}`);
    if (cursor) cursor.style.left = `${pct}%`;

    const timeDisplay = document.querySelector(`#${cardId} .card-playback-time`);
    if (timeDisplay) {
      timeDisplay.textContent = `${this.formatTime(elapsed)} / ${this.formatTime(duration)}`;
    }

    this.currentAnimFrame = requestAnimationFrame(() => this.updatePlaybackCursor(cardId));
  }

  handleCanvasClick(cardId, event) {
    const audioBuffer = this.audioBuffers.get(cardId);
    if (!audioBuffer) return;

    const canvas = document.getElementById(`canvas-${cardId}`);
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const pct = (event.clientX - rect.left) / rect.width;

    // If playing this card, restart from seek position
    if (this.currentlyPlaying === cardId) {
      this.stopCurrentAudio();
    }

    // Show cursor at click position
    const cursor = document.getElementById(`cursor-${cardId}`);
    if (cursor) {
      cursor.classList.add('visible');
      cursor.style.left = `${pct * 100}%`;
    }
  }

  // ── Summary stats ────────────────────────────────────────
  updateSummary(total) {
    this.summaryProcessed.textContent = `${this.processedCount} / ${total}`;

    if (this.wcsrValues.length > 0) {
      const avg = this.wcsrValues.reduce((a, b) => a + b, 0) / this.wcsrValues.length;
      const best = Math.max(...this.wcsrValues);
      const worst = Math.min(...this.wcsrValues);

      this.summaryAvgWcsr.textContent = `${(avg * 100).toFixed(1)}%`;
      this.summaryBest.textContent = `${(best * 100).toFixed(1)}%`;
      this.summaryWorst.textContent = `${(worst * 100).toFixed(1)}%`;
    }
  }

  // ── Progress ─────────────────────────────────────────────
  updateProgress(percent, text) {
    this.progressSection.style.display = 'block';
    this.progressFill.style.width = `${percent}%`;
    this.progressPercent.textContent = `${Math.round(percent)}%`;
    this.progressText.textContent = text;
  }

  // ── Utilities ────────────────────────────────────────────
  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
  }

  esc(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  window.app = new BulkValidationApp();
});
