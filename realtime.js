/**
 * Real-time Chord Detection
 * 
 * Captures microphone input using AudioWorklet for low-latency processing.
 * Sends audio to worker for onset detection and chord classification.
 */

import { CONFIG } from './modules/config.js';
import { ClassificationService } from './modules/classification-service.js';

class RealtimeChordDetector {
  constructor() {
    // Audio state
    this.audioContext = null;
    this.mediaStream = null;
    this.workletNode = null;
    this.sourceNode = null;
    this.isRecording = false;
    this.startTime = 0;

    // Classification service
    this.classificationService = new ClassificationService({
      model: CONFIG.classification.model,
      onProgress: (percent, message) => {
        this.updateStatus(message);
      },
      onResult: (result, cqtData) => {
        this.handleClassificationResult(result, cqtData);
      }
    });

    // Detection history
    this.history = [];
    this.maxHistory = 50;

    this.initElements();
    this.bindEvents();
    this.init();
  }

  initElements() {
    // Controls
    this.inputDeviceSelect = document.getElementById('inputDevice');
    this.recordBtn = document.getElementById('recordBtn');
    this.clearHistoryBtn = document.getElementById('clearHistory');

    // Parameters
    this.onsetThresholdInput = document.getElementById('onsetThreshold');
    this.minOnsetIntervalInput = document.getElementById('minOnsetInterval');
    this.windowSizeInput = document.getElementById('windowSize');
    this.ignoreSubsequentInput = document.getElementById('ignoreSubsequent');
    this.flexibleWindowInput = document.getElementById('flexibleWindow');

    // Mutual exclusion: flexible window disables ignore subsequent
    this.flexibleWindowInput.addEventListener('change', () => {
      this.ignoreSubsequentInput.disabled = this.flexibleWindowInput.checked;
    });

    // Status
    this.statusIndicator = document.getElementById('statusIndicator');
    this.statusDot = this.statusIndicator.querySelector('.status-dot');
    this.statusText = this.statusIndicator.querySelector('.status-text');

    // Display
    this.currentChordEl = document.getElementById('currentChord');
    this.chordConfidenceEl = document.getElementById('chordConfidence');
    this.historyList = document.getElementById('historyList');

    // CQT Visualization
    this.cqtCanvas = document.getElementById('cqtCanvas');
    this.cqtCtx = this.cqtCanvas.getContext('2d');
    this.cqtInfo = document.getElementById('cqtInfo');

    // Pre-compute CQT colormap (viridis-like)
    this.cqtColormap = this.generateColormap();
  }

  /**
   * Generate a viridis-like colormap for CQT visualization
   */
  generateColormap() {
    const colors = [];
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      // Viridis-inspired colors
      const r = Math.floor(255 * Math.min(1, 0.267 + 2.2 * t - 1.8 * t * t));
      const g = Math.floor(255 * (0.004 + 1.0 * t - 0.35 * t * t));
      const b = Math.floor(255 * Math.max(0, 0.329 + 1.2 * t - 2.0 * t * t + 0.8 * t * t * t));
      colors.push([r, g, b]);
    }
    return colors;
  }

  bindEvents() {
    this.recordBtn.addEventListener('click', () => this.toggleRecording());
    this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
    this.inputDeviceSelect.addEventListener('change', () => this.handleDeviceChange());
  }

  async init() {
    try {
      this.updateStatus('Initializing...');

      // Get audio devices
      await this.loadAudioDevices();

      // Initialize classification service
      await this.classificationService.init();

      this.updateStatus('Ready');
      this.statusDot.classList.add('ready');
      this.recordBtn.disabled = false;

    } catch (error) {
      console.error('Initialization error:', error);
      this.updateStatus('Error: ' + error.message);
    }
  }

  async loadAudioDevices() {
    try {
      // Request permission first
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());

      // Get device list
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter(d => d.kind === 'audioinput');

      this.inputDeviceSelect.innerHTML = '';
      audioInputs.forEach((device, i) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.textContent = device.label || `Microphone ${i + 1}`;
        this.inputDeviceSelect.appendChild(option);
      });

    } catch (error) {
      console.error('Failed to get audio devices:', error);
      this.inputDeviceSelect.innerHTML = '<option value="">No microphone access</option>';
    }
  }

  handleDeviceChange() {
    // If recording, restart with new device
    if (this.isRecording) {
      this.stopRecording();
      this.startRecording();
    }
  }

  async toggleRecording() {
    if (this.isRecording) {
      await this.stopRecording();
    } else {
      await this.startRecording();
    }
  }

  async startRecording() {
    try {
      this.updateStatus('Starting microphone...');

      // Get selected device
      const deviceId = this.inputDeviceSelect.value;
      const constraints = {
        audio: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      };

      // Get media stream
      this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

      // Create audio context
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const sampleRate = this.audioContext.sampleRate;

      console.log(`Audio context sample rate: ${sampleRate}`);

      // Create source node
      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

      // Try to use AudioWorklet, fallback to ScriptProcessor
      try {
        await this.setupAudioWorklet(sampleRate);
      } catch (workletError) {
        console.warn('AudioWorklet not available, using ScriptProcessor:', workletError);
        this.setupScriptProcessor(sampleRate);
      }

      // Start streaming in worker
      this.classificationService.worker.postMessage({
        type: 'start-stream',
        payload: {
          sampleRate: sampleRate,
          windowSize: parseFloat(this.windowSizeInput.value),
          threshold: parseFloat(this.onsetThresholdInput.value),
          minInterval: parseInt(this.minOnsetIntervalInput.value),
          ignoreSubsequent: this.ignoreSubsequentInput.checked,
          flexibleWindow: this.flexibleWindowInput.checked
        }
      });

      this.isRecording = true;
      this.startTime = Date.now();

      // Update UI
      this.recordBtn.classList.add('recording');
      this.recordBtn.querySelector('.btn-text').textContent = 'Stop Recording';
      this.recordBtn.querySelector('.btn-icon').textContent = '⏹️';
      this.statusDot.classList.remove('ready');
      this.statusDot.classList.add('recording');
      this.updateStatus('Recording...');

    } catch (error) {
      console.error('Failed to start recording:', error);
      this.updateStatus('Error: ' + error.message);
    }
  }

  async setupAudioWorklet(sampleRate) {
    // Load worklet module
    await this.audioContext.audioWorklet.addModule('./modules/audio-capture-processor.js');

    // Create worklet node
    this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-capture-processor', {
      processorOptions: {
        bufferSize: 4096,
        sampleRate: sampleRate
      }
    });

    // Handle messages from worklet
    this.workletNode.port.onmessage = (event) => {
      if (event.data.type === 'audio' && this.isRecording) {
        this.sendAudioToWorker(event.data.samples);
      }
    };

    // Connect nodes
    this.sourceNode.connect(this.workletNode);
    // Don't connect to destination to avoid feedback
  }

  setupScriptProcessor(sampleRate) {
    // Fallback for browsers without AudioWorklet
    const bufferSize = 4096;
    this.scriptNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

    this.scriptNode.onaudioprocess = (event) => {
      if (this.isRecording) {
        const samples = event.inputBuffer.getChannelData(0);
        this.sendAudioToWorker(new Float32Array(samples));
      }
    };

    this.sourceNode.connect(this.scriptNode);
    this.scriptNode.connect(this.audioContext.destination);
  }

  sendAudioToWorker(samples) {
    // Send to worker for onset detection
    this.classificationService.worker.postMessage({
      type: 'stream-detect',
      payload: {
        audioData: samples,
        timestamp: Date.now() - this.startTime
      }
    });
  }

  async stopRecording() {
    // Stop streaming in worker
    this.classificationService.worker.postMessage({ type: 'stop-stream' });

    // Disconnect and clean up
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    if (this.scriptNode) {
      this.scriptNode.disconnect();
      this.scriptNode = null;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    this.isRecording = false;

    // Update UI
    this.recordBtn.classList.remove('recording');
    this.recordBtn.querySelector('.btn-text').textContent = 'Start Recording';
    this.recordBtn.querySelector('.btn-icon').textContent = '🎙️';
    this.statusDot.classList.remove('recording');
    this.statusDot.classList.add('ready');
    this.updateStatus('Ready');
  }

  handleClassificationResult(result, cqtData) {
    // Update current chord display
    this.currentChordEl.textContent = result.mirexChord || result.chord;
    this.currentChordEl.classList.add('detected');
    setTimeout(() => this.currentChordEl.classList.remove('detected'), 300);

    const confidence = (result.confidence * 100).toFixed(1);
    this.chordConfidenceEl.textContent = `${confidence}% confidence`;

    // Render CQT visualization
    if (cqtData) {
      this.renderCQT(cqtData);
    }

    // Add to history
    this.addToHistory(result);
  }

  /**
   * Render CQT spectrogram on canvas
   * Uses ImageData for efficient rendering
   */
  renderCQT(cqtData) {
    const { data, bins, frames } = cqtData;

    // Update info label
    this.cqtInfo.textContent = `${bins} bins × ${frames} frames`;

    // Get canvas dimensions
    const canvasWidth = this.cqtCanvas.width;
    const canvasHeight = this.cqtCanvas.height;

    // Create image data
    const imageData = this.cqtCtx.createImageData(canvasWidth, canvasHeight);
    const pixels = imageData.data;

    // Scale factors
    const xScale = frames / canvasWidth;
    const yScale = bins / canvasHeight;

    // Fill pixels
    for (let y = 0; y < canvasHeight; y++) {
      for (let x = 0; x < canvasWidth; x++) {
        // Map canvas coordinates to CQT coordinates
        const frame = Math.floor(x * xScale);
        const bin = bins - 1 - Math.floor(y * yScale); // Flip Y axis (low freq at bottom)

        // Get CQT value (data is in bins × frames format, row-major)
        const idx = bin * frames + frame;
        const value = Math.min(1, Math.max(0, data[idx] || 0));

        // Map to color
        const colorIdx = Math.floor(value * 255);
        const [r, g, b] = this.cqtColormap[colorIdx];

        // Set pixel
        const pixelIdx = (y * canvasWidth + x) * 4;
        pixels[pixelIdx] = r;
        pixels[pixelIdx + 1] = g;
        pixels[pixelIdx + 2] = b;
        pixels[pixelIdx + 3] = 255;
      }
    }

    // Draw to canvas
    this.cqtCtx.putImageData(imageData, 0, 0);
  }

  addToHistory(result) {
    const timestamp = new Date().toLocaleTimeString();
    const entry = {
      chord: result.mirexChord || result.chord,
      confidence: result.confidence,
      timestamp: timestamp
    };

    this.history.unshift(entry);
    if (this.history.length > this.maxHistory) {
      this.history.pop();
    }

    this.renderHistory();
  }

  renderHistory() {
    if (this.history.length === 0) {
      this.historyList.innerHTML = '<div class="history-empty">No chords detected yet. Start recording to begin.</div>';
      return;
    }

    this.historyList.innerHTML = this.history.map(entry => `
            <div class="history-item">
                <div class="history-chord">${entry.chord}</div>
                <div class="history-details">
                    <div class="history-time">${entry.timestamp}</div>
                    <div class="history-confidence">${(entry.confidence * 100).toFixed(1)}% confidence</div>
                </div>
            </div>
        `).join('');
  }

  clearHistory() {
    this.history = [];
    this.currentChordEl.textContent = '—';
    this.chordConfidenceEl.textContent = '';
    this.renderHistory();
  }

  updateStatus(message) {
    this.statusText.textContent = message;
  }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  window.detector = new RealtimeChordDetector();
});
