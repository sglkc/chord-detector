/**
 * Single Chord Classifier
 * Classify individual chord samples from multiple audio files
 * All heavy processing happens in Web Worker
 */

import { CONFIG } from './modules/config.js';
import { ClassificationService } from './modules/classification-service.js';

class SingleChordClassifier {
  constructor() {
    this.files = [];
    this.audioBuffers = new Map();
    this.results = new Map();
    this.audioContext = null;
    this.currentlyPlaying = null;
    this.currentSource = null;

    // Current model
    this.currentModelName = 'graph';

    // Classification service - handles all heavy processing in worker
    this.classificationService = new ClassificationService({
      model: this.currentModelName,
      onProgress: (percent, message) => {
        this.updateProgress(percent, message);
      }
    });

    this.initElements();
    this.bindEvents();
  }

  initElements() {
    // Upload elements
    this.uploadZone = document.getElementById('uploadZone');
    this.audioFilesInput = document.getElementById('audioFiles');
    this.fileCount = document.getElementById('fileCount');
    this.classifyBtn = document.getElementById('classifyBtn');
    this.clearBtn = document.getElementById('clearBtn');

    // Progress elements
    this.progressSection = document.getElementById('progressSection');
    this.progressFill = document.getElementById('progressFill');
    this.progressText = document.getElementById('progressText');
    this.progressPercent = document.getElementById('progressPercent');

    // Results elements
    this.resultsSection = document.getElementById('resultsSection');
    this.resultsGrid = document.getElementById('resultsGrid');
    this.processedCount = document.getElementById('processedCount');
  }

  bindEvents() {
    // File input
    this.audioFilesInput.addEventListener('change', (e) => this.handleFileSelect(e));

    // Drag and drop
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
        this.handleFiles(Array.from(e.dataTransfer.files));
      }
    });

    // Buttons
    this.classifyBtn.addEventListener('click', () => this.classifyAll());
    this.clearBtn.addEventListener('click', () => this.clearAll());

    // CQT backend selection
    this.cqtBackendSelect?.addEventListener('change', () => this.handleCqtBackendChange());
  }

  handleFileSelect(event) {
    const files = Array.from(event.target.files);
    this.handleFiles(files);
  }

  handleFiles(files) {
    // Filter audio files
    const audioFiles = files.filter(file => file.type.startsWith('audio/'));

    if (audioFiles.length === 0) {
      alert('Please select audio files only.');
      return;
    }

    this.files = [...this.files, ...audioFiles];
    this.updateFileCount();
    this.classifyBtn.disabled = false;
    this.clearBtn.disabled = false;
  }

  updateFileCount() {
    const count = this.files.length;
    this.fileCount.textContent = `${count} file${count !== 1 ? 's' : ''} selected`;
  }

  clearAll() {
    this.files = [];
    this.audioBuffers.clear();
    this.results.clear();
    this.audioFilesInput.value = '';
    this.updateFileCount();
    this.classifyBtn.disabled = true;
    this.clearBtn.disabled = true;
    this.resultsGrid.innerHTML = '';
    this.resultsSection.style.display = 'none';
    this.stopCurrentAudio();
  }

  async classifyAll() {
    if (this.files.length === 0) return;

    // Show progress
    this.progressSection.style.display = 'block';
    this.resultsSection.style.display = 'block';
    this.resultsGrid.innerHTML = '';
    this.classifyBtn.disabled = true;

    try {
      // Initialize components
      this.updateProgress(5, 'Initializing...');
      await this.initializeComponents();

      const totalFiles = this.files.length;

      for (let i = 0; i < totalFiles; i++) {
        const file = this.files[i];
        const progressPercent = 10 + (i / totalFiles) * 85;

        this.updateProgress(progressPercent, `Processing ${file.name} (${i + 1}/${totalFiles})`);

        // Create card placeholder
        const cardId = `card-${i}`;
        this.createCardPlaceholder(cardId, file.name);

        try {
          // Load audio
          const audioBuffer = await this.loadAudioFile(file);
          this.audioBuffers.set(cardId, audioBuffer);

          // Get audio data for classification
          const audioData = audioBuffer.getChannelData(0);

          // Classify using ClassificationService (worker handles CQT + prediction)
          const { prediction, cqtData } = await this.classificationService.classify(audioData);
          this.results.set(cardId, prediction);

          // Update card with result and CQT visualization
          this.updateCard(cardId, file, audioBuffer, prediction, cqtData);
        } catch (error) {
          console.error(`Error processing ${file.name}:`, error);
          this.updateCardError(cardId, file.name, error.message);
        }

        this.processedCount.textContent = `${i + 1} processed`;

        // Yield to keep UI responsive
        await new Promise(resolve => setTimeout(resolve, 0));
      }

      this.updateProgress(100, 'Classification complete!');

      // Hide progress after a short delay
      setTimeout(() => {
        this.progressSection.style.display = 'none';
      }, 1500);

    } catch (error) {
      console.error('Classification error:', error);
      this.updateProgress(0, `Error: ${error.message}`);
    } finally {
      this.classifyBtn.disabled = false;
    }
  }

  async initializeComponents() {
    // Initialize audio context
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    // Use model from config (graph-v2)
    const selectedModel = CONFIG.classification.model;

    // Update classification service if model changed
    if (this.currentModelName !== selectedModel) {
      this.currentModelName = selectedModel;
      await this.classificationService.setModel(selectedModel);
      console.log(`Model updated: ${selectedModel}`);
    }

    // Initialize the classification service (loads model if needed)
    await this.classificationService.init();
  }

  async loadAudioFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const audioBuffer = await this.audioContext.decodeAudioData(e.target.result);
          resolve(audioBuffer);
        } catch (error) {
          reject(new Error('Failed to decode audio file'));
        }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsArrayBuffer(file);
    });
  }

  createCardPlaceholder(cardId, fileName) {
    const card = document.createElement('div');
    card.id = cardId;
    card.className = 'result-card loading';
    card.innerHTML = `
            <div class="card-header">
                <div class="file-info">
                    <div class="file-name">${this.escapeHtml(fileName)}</div>
                    <div class="file-duration">Loading...</div>
                </div>
            </div>
            <div class="spectrogram-container">
                <div class="loading-spinner"></div>
            </div>
            <div class="classification-result">
                <div class="loading-placeholder">Analyzing...</div>
            </div>
        `;
    this.resultsGrid.appendChild(card);
  }

  updateCard(cardId, file, audioBuffer, result, cqtData) {
    const card = document.getElementById(cardId);
    if (!card) return;

    card.className = 'result-card';

    const duration = audioBuffer.duration.toFixed(2);
    const confidence = (result.confidence * 100).toFixed(1);
    const confidenceClass = result.confidence >= 0.7 ? 'high' : result.confidence >= 0.4 ? 'medium' : 'low';

    // Get top 3 predictions
    const topPreds = result.topPredictions.slice(0, 3);

    card.innerHTML = `
            <div class="card-header">
                <div class="file-info">
                    <div class="file-name" title="${this.escapeHtml(file.name)}">${this.escapeHtml(file.name)}</div>
                    <div class="file-duration">${duration}s</div>
                </div>
                <button class="play-btn" data-card-id="${cardId}" title="Play audio">
                    <span class="play-icon">▶</span>
                </button>
            </div>
            <div class="spectrogram-container">
                <canvas class="spectrogram-canvas" id="canvas-${cardId}"></canvas>
                <div class="spectrogram-overlay"></div>
            </div>
            <div class="classification-result">
                <div class="chord-prediction">
                    <div class="chord-badge">${this.escapeHtml(result.mirexChord)}</div>
                    <div class="confidence-info">
                        <div class="confidence-label">Confidence</div>
                        <div class="confidence-value ${confidenceClass}">${confidence}%</div>
                        <div class="confidence-bar-container">
                            <div class="confidence-bar">
                                <div class="confidence-fill ${confidenceClass}" style="width: ${confidence}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="top-predictions">
                    <div class="top-predictions-label">Top Predictions</div>
                    <div class="prediction-list">
                        ${topPreds.map(pred => `
                            <div class="prediction-item">
                                <span class="prediction-chord">${this.escapeHtml(pred.mirexChord)}</span>
                                <span class="prediction-prob">${(pred.probability * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;

    // Draw spectrogram
    this.drawSpectrogram(cardId, cqtData);

    // Add play button event
    const playBtn = card.querySelector('.play-btn');
    playBtn.addEventListener('click', () => this.togglePlay(cardId));
  }

  updateCardError(cardId, fileName, errorMessage) {
    const card = document.getElementById(cardId);
    if (!card) return;

    card.className = 'result-card error';
    card.innerHTML = `
            <div class="card-header">
                <div class="file-info">
                    <div class="file-name">${this.escapeHtml(fileName)}</div>
                    <div class="file-duration">Error</div>
                </div>
            </div>
            <div class="spectrogram-container" style="height: 136px; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 2rem;">⚠️</span>
            </div>
            <div class="classification-result">
                <div class="loading-placeholder" style="color: var(--accent-error);">
                    ${this.escapeHtml(errorMessage)}
                </div>
            </div>
        `;
  }

  drawSpectrogram(cardId, cqtData) {
    const canvas = document.getElementById(`canvas-${cardId}`);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.offsetWidth || 300;
    const height = canvas.offsetHeight || 120;

    canvas.width = width;
    canvas.height = height;

    // Handle case when cqtData is not available
    if (!cqtData || !cqtData.magnitudes) {
      // Draw a placeholder gradient
      const gradient = ctx.createLinearGradient(0, height, 0, 0);
      gradient.addColorStop(0, '#1a1a2e');
      gradient.addColorStop(0.5, '#16213e');
      gradient.addColorStop(1, '#0f3460');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      // Add text indicating no spectrogram
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.font = '12px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('CQT Visualization', width / 2, height / 2);
      return;
    }

    const { magnitudes, numFrames, numBins } = cqtData;

    // Normalize magnitudes
    let maxVal = 0;
    for (const frame of magnitudes) {
      for (const val of frame) {
        if (val > maxVal) maxVal = val;
      }
    }

    // Draw spectrogram
    const frameWidth = width / numFrames;
    const binHeight = height / numBins;

    for (let t = 0; t < numFrames; t++) {
      for (let b = 0; b < numBins; b++) {
        const value = maxVal > 0 ? magnitudes[t][b] / maxVal : 0;
        const color = this.viridisColor(value);
        ctx.fillStyle = color;
        ctx.fillRect(
          t * frameWidth,
          (numBins - 1 - b) * binHeight,
          Math.ceil(frameWidth) + 1,
          Math.ceil(binHeight) + 1
        );
      }
    }
  }

  viridisColor(value) {
    const v = Math.max(0, Math.min(1, value));

    const r = Math.round(255 * (0.267004 + v * (0.329415 + v * (-0.508378 + v * 1.137680))));
    const g = Math.round(255 * (0.004874 + v * (0.873158 + v * (-0.058404 + v * -0.322897))));
    const b = Math.round(255 * (0.329415 + v * (0.280197 + v * (-1.314181 + v * 1.171356))));

    return `rgb(${Math.max(0, Math.min(255, r))}, ${Math.max(0, Math.min(255, g))}, ${Math.max(0, Math.min(255, b))})`;
  }

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

    // Create source
    this.currentSource = this.audioContext.createBufferSource();
    this.currentSource.buffer = audioBuffer;
    this.currentSource.connect(this.audioContext.destination);

    // Update UI
    const playBtn = document.querySelector(`#${cardId} .play-btn`);
    if (playBtn) {
      playBtn.classList.add('playing');
      playBtn.querySelector('.play-icon').textContent = '⏹';
    }

    this.currentlyPlaying = cardId;

    // Handle playback end
    this.currentSource.onended = () => {
      if (this.currentlyPlaying === cardId) {
        this.stopCurrentAudio();
      }
    };

    this.currentSource.start(0);
  }

  stopCurrentAudio() {
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch (e) {
        // Already stopped
      }
      this.currentSource.disconnect();
      this.currentSource = null;
    }

    if (this.currentlyPlaying) {
      const playBtn = document.querySelector(`#${this.currentlyPlaying} .play-btn`);
      if (playBtn) {
        playBtn.classList.remove('playing');
        playBtn.querySelector('.play-icon').textContent = '▶';
      }
      this.currentlyPlaying = null;
    }
  }

  updateProgress(percent, text) {
    this.progressFill.style.width = `${percent}%`;
    this.progressPercent.textContent = `${Math.round(percent)}%`;
    this.progressText.textContent = text;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.app = new SingleChordClassifier();
});
