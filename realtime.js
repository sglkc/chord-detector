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
      model: 'graph',
      onProgress: (percent, message) => {
        this.updateStatus(message);
      },
      onResult: (result) => {
        this.handleClassificationResult(result);
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
    this.modelSelect = document.getElementById('modelSelect');
    this.recordBtn = document.getElementById('recordBtn');
    this.clearHistoryBtn = document.getElementById('clearHistory');

    // Parameters
    this.onsetThresholdInput = document.getElementById('onsetThreshold');
    this.minOnsetIntervalInput = document.getElementById('minOnsetInterval');
    this.windowSizeInput = document.getElementById('windowSize');
    this.ignoreSubsequentInput = document.getElementById('ignoreSubsequent');

    // Status
    this.statusIndicator = document.getElementById('statusIndicator');
    this.statusDot = this.statusIndicator.querySelector('.status-dot');
    this.statusText = this.statusIndicator.querySelector('.status-text');

    // Display
    this.currentChordEl = document.getElementById('currentChord');
    this.chordConfidenceEl = document.getElementById('chordConfidence');
    this.historyList = document.getElementById('historyList');
  }

  bindEvents() {
    this.recordBtn.addEventListener('click', () => this.toggleRecording());
    this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
    this.modelSelect.addEventListener('change', () => this.handleModelChange());
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

  async handleModelChange() {
    const newModel = this.modelSelect.value;
    await this.classificationService.setModel(newModel);
    console.log(`Model changed to: ${newModel}`);
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
          ignoreSubsequent: this.ignoreSubsequentInput.checked
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

  handleClassificationResult(result) {
    // Update current chord display
    this.currentChordEl.textContent = result.mirexChord || result.chord;
    this.currentChordEl.classList.add('detected');
    setTimeout(() => this.currentChordEl.classList.remove('detected'), 300);

    const confidence = (result.confidence * 100).toFixed(1);
    this.chordConfidenceEl.textContent = `${confidence}% confidence`;

    // Add to history
    this.addToHistory(result);
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
