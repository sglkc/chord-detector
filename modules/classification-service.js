/**
 * Classification Service
 * 
 * A modular, reusable service for chord classification that can be used
 * across different pages and supports multiple modes:
 * - Batch mode: Process multiple audio segments (validation, multi-file)
 * - Single-shot mode: Classify a single audio segment
 * - Real-time streaming mode: Continuous classification from microphone
 * 
 * Features:
 * - Web Worker support for non-blocking UI
 * - Pre-initialization for low-latency real-time use
 * - Transferable ArrayBuffers for minimal copy overhead
 * - Fallback to main-thread processing if worker fails
 * 
 * Usage:
 *   import { ClassificationService } from './modules/classification-service.js';
 *   
 *   const service = new ClassificationService({
 *       model: 'graph-v2',
 *       cqtBackend: 'wasm',
 *       onProgress: (percent, message) => updateUI(percent, message)
 *   });
 *   
 *   await service.init();
 *   const result = await service.classify(audioData);
 *   const results = await service.classifyBatch(onsets, audioData, sampleRate, duration);
 */

import { ChordClassifier } from './chord-classifier.js';
import { CQTExtractor } from './cqt-extractor.js';
import { CONFIG } from './config.js';

/**
 * Classification modes
 */
export const ClassificationMode = {
  WORKER: 'worker',           // Use Web Worker (non-blocking)
  MAIN_THREAD: 'main-thread', // Use main thread (blocking but simpler)
  AUTO: 'auto'                // Try worker, fallback to main thread
};

/**
 * Classification Service - Modular chord classification
 */
export class ClassificationService {
  /**
   * Create a new ClassificationService
   * @param {Object} options - Configuration options
   * @param {string} options.model - Model name ('graph', 'graph-v2', 'layers')
   * @param {string} options.cqtBackend - CQT backend ('wasm', 'librosa', 'showcqt')
   * @param {string} options.mode - Classification mode ('worker', 'main-thread', 'auto')
   * @param {Function} options.onProgress - Progress callback (percent, message)
   * @param {Function} options.onResult - Real-time result callback (result)
   * @param {Object} options.config - Override CONFIG settings
   */
  constructor(options = {}) {
    this.options = {
      model: options.model || CONFIG.classification.model || 'graph-v2',
      cqtBackend: options.cqtBackend || CONFIG.classification.cqtBackend || 'wasm',
      mode: options.mode || ClassificationMode.AUTO,
      onProgress: options.onProgress || (() => { }),
      onResult: options.onResult || (() => { }),
      config: options.config || CONFIG
    };

    // State
    this.initialized = false;
    this.worker = null;
    this.workerReady = false;
    this.useWorker = false;

    // Main-thread fallback components
    this.classifier = null;
    this.cqtExtractor = null;

    // Worker communication
    this.pendingCallbacks = new Map();
    this.messageId = 0;

    // Real-time streaming state
    this.isStreaming = false;
    this.streamBuffer = null;
    this.streamBufferSize = 0;
  }

  /**
   * Initialize the service (load model, warm up worker)
   * Call this once on page load for lowest latency
   */
  async init() {
    if (this.initialized) {
      console.log('[ClassificationService] Already initialized');
      return;
    }

    const { mode, model, cqtBackend } = this.options;

    // Determine if we should use worker
    this.useWorker = mode === ClassificationMode.WORKER || mode === ClassificationMode.AUTO;

    if (this.useWorker) {
      try {
        await this._initWorker();
        console.log('[ClassificationService] Worker initialized');
      } catch (error) {
        console.warn('[ClassificationService] Worker init failed, using main thread:', error);
        if (mode === ClassificationMode.WORKER) {
          throw error; // User explicitly requested worker
        }
        this.useWorker = false;
      }
    }

    // Initialize main-thread components as fallback or primary
    if (!this.useWorker || mode === ClassificationMode.AUTO) {
      await this._initMainThread();
      console.log('[ClassificationService] Main thread components initialized');
    }

    this.initialized = true;
    console.log(`[ClassificationService] Ready (mode: ${this.useWorker ? 'worker' : 'main-thread'})`);
  }

  /**
   * Initialize the Web Worker
   */
  async _initWorker() {
    return new Promise((resolve, reject) => {
      try {
        this.worker = new Worker('./modules/classification-worker.js');

        this.worker.onmessage = (event) => this._handleWorkerMessage(event.data);
        this.worker.onerror = (error) => {
          console.error('[ClassificationService] Worker error:', error);
          reject(error);
        };

        // Send init message
        const id = ++this.messageId;
        const modelPath = `${window.location.origin}/models/${this.options.model}/model.json`;

        this.pendingCallbacks.set(id, { resolve, reject });

        this.worker.postMessage({
          type: 'init',
          id,
          payload: {
            modelPath,
            sampleRate: this.options.config.audio.sampleRate,
            config: this.options.config
          }
        });

        // Timeout after 30 seconds
        setTimeout(() => {
          if (this.pendingCallbacks.has(id)) {
            this.pendingCallbacks.delete(id);
            reject(new Error('Worker initialization timeout'));
          }
        }, 30000);

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Initialize main-thread components
   */
  async _initMainThread() {
    const { model, cqtBackend, config } = this.options;

    // Initialize CQT extractor
    this.cqtExtractor = new CQTExtractor(cqtBackend);
    await this.cqtExtractor.init(config.audio.sampleRate);

    // Initialize classifier and load model
    this.classifier = new ChordClassifier();
    const modelPath = `./models/${model}/model.json`;
    await this.classifier.loadModel(modelPath);
  }

  /**
   * Handle messages from the worker
   */
  _handleWorkerMessage(data) {
    const { type, id, ...rest } = data;

    switch (type) {
      case 'ready':
        this.workerReady = true;
        if (this.pendingCallbacks.has(id)) {
          this.pendingCallbacks.get(id).resolve();
          this.pendingCallbacks.delete(id);
        }
        break;

      case 'progress':
        this.options.onProgress(rest.percent, `Classifying... (${rest.current}/${rest.total})`);
        break;

      case 'result':
        if (this.pendingCallbacks.has(id)) {
          this.pendingCallbacks.get(id).resolve(rest.predictions || rest.prediction);
          this.pendingCallbacks.delete(id);
        }
        break;

      case 'stream-result':
        // Real-time streaming result
        this.options.onResult(rest.prediction);
        break;

      case 'error':
        console.error('[ClassificationService] Worker error:', rest.error);
        if (this.pendingCallbacks.has(id)) {
          this.pendingCallbacks.get(id).reject(new Error(rest.error));
          this.pendingCallbacks.delete(id);
        }
        break;

      case 'status':
        console.log('[ClassificationService] Worker status:', rest.message);
        break;
    }
  }

  /**
   * Send a message to the worker and wait for response
   */
  _sendWorkerMessage(type, payload, transferables = []) {
    return new Promise((resolve, reject) => {
      const id = ++this.messageId;
      this.pendingCallbacks.set(id, { resolve, reject });
      this.worker.postMessage({ type, id, payload }, transferables);
    });
  }

  /**
   * Classify a single audio segment
   * @param {Float32Array} audioData - Audio samples
   * @param {number} sampleRate - Sample rate (optional, uses config default)
   * @returns {Object} Classification result
   */
  async classify(audioData, sampleRate = null) {
    if (!this.initialized) {
      await this.init();
    }

    const sr = sampleRate || this.options.config.audio.sampleRate;

    if (this.useWorker && this.workerReady) {
      try {
        // Use transferable for zero-copy (if possible)
        const buffer = audioData.buffer.slice(0);
        const result = await this._sendWorkerMessage('classify-single', {
          audioData: new Float32Array(buffer),
          sampleRate: sr
        });
        return result;
      } catch (error) {
        console.warn('[ClassificationService] Worker classify failed, using main thread:', error);
      }
    }

    // Main thread fallback
    return this._classifyMainThread(audioData, sr);
  }

  /**
   * Classify using main thread
   */
  async _classifyMainThread(audioData, sampleRate) {
    const config = this.options.config;

    // Extract CQT features
    const features = await this.cqtExtractor.extractFeatures(audioData, config);

    // Classify
    const result = await this.classifier.predict(features);

    return {
      chord: result.chord,
      mirexChord: result.mirexChord,
      confidence: result.confidence,
      topPredictions: result.topPredictions
    };
  }

  /**
   * Classify multiple audio segments (batch mode)
   * @param {Array} onsets - Array of onset objects with 'time' property
   * @param {Float32Array} audioData - Full audio data
   * @param {number} sampleRate - Sample rate
   * @param {number} audioDuration - Total audio duration
   * @returns {Array} Array of prediction results
   */
  async classifyBatch(onsets, audioData, sampleRate, audioDuration) {
    if (!this.initialized) {
      await this.init();
    }

    if (this.useWorker && this.workerReady) {
      try {
        const result = await this._sendWorkerMessage('classify', {
          onsets,
          audioData: audioData,
          sampleRate,
          windowSize: this.options.config.classification.windowSize,
          audioDuration
        });
        return result;
      } catch (error) {
        console.warn('[ClassificationService] Worker batch failed, using main thread:', error);
      }
    }

    // Main thread fallback with progress updates
    return this._classifyBatchMainThread(onsets, audioData, sampleRate, audioDuration);
  }

  /**
   * Batch classify using main thread with periodic yields for UI
   */
  async _classifyBatchMainThread(onsets, audioData, sampleRate, audioDuration) {
    const predictions = [];
    const config = this.options.config;
    const windowSamples = Math.floor(config.classification.windowSize * sampleRate);

    for (let i = 0; i < onsets.length; i++) {
      const onset = onsets[i];
      const startSample = Math.floor(onset.time * sampleRate);
      const endSample = Math.min(startSample + windowSamples, audioData.length);

      // Determine end time
      let endTime;
      if (i < onsets.length - 1) {
        endTime = onsets[i + 1].time;
      } else {
        endTime = audioDuration;
      }

      // Extract window
      const windowData = audioData.slice(startSample, endSample);

      if (windowData.length < windowSamples * 0.5) {
        continue;
      }

      // Extract features and classify
      const features = await this.cqtExtractor.extractFeatures(windowData, config);
      const result = await this.classifier.predict(features);

      predictions.push({
        start: onset.time,
        end: endTime,
        chord: result.chord,
        confidence: result.confidence,
        mirexChord: result.mirexChord
      });

      // Progress update and yield every 2 items
      if (i % 2 === 0) {
        const percent = Math.round(((i + 1) / onsets.length) * 100);
        this.options.onProgress(percent, `Classifying... (${i + 1}/${onsets.length})`);
        // Yield to event loop
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    return predictions;
  }

  /**
   * Start real-time streaming classification
   * @param {Function} onResult - Callback for each classification result
   * @param {Object} options - Streaming options
   */
  startStream(onResult = null, options = {}) {
    if (this.isStreaming) {
      console.warn('[ClassificationService] Already streaming');
      return;
    }

    this.isStreaming = true;
    this.streamBuffer = new Float32Array(options.bufferSize || 96000); // 2 seconds at 48kHz
    this.streamBufferSize = 0;

    if (onResult) {
      this.options.onResult = onResult;
    }

    if (this.useWorker && this.workerReady) {
      this.worker.postMessage({
        type: 'start-stream',
        payload: {
          sampleRate: this.options.config.audio.sampleRate,
          windowSize: this.options.config.classification.windowSize
        }
      });
    }

    console.log('[ClassificationService] Streaming started');
  }

  /**
   * Feed audio samples to the stream
   * @param {Float32Array} samples - Audio samples to process
   */
  async feedAudio(samples) {
    if (!this.isStreaming) {
      console.warn('[ClassificationService] Not streaming');
      return;
    }

    // Append to buffer
    if (this.streamBufferSize + samples.length > this.streamBuffer.length) {
      // Shift buffer to make room
      const shift = samples.length;
      this.streamBuffer.copyWithin(0, shift);
      this.streamBufferSize = Math.max(0, this.streamBufferSize - shift);
    }

    this.streamBuffer.set(samples, this.streamBufferSize);
    this.streamBufferSize += samples.length;

    // Check if we have enough samples for classification
    const windowSamples = Math.floor(
      this.options.config.classification.windowSize *
      this.options.config.audio.sampleRate
    );

    if (this.streamBufferSize >= windowSamples) {
      // Extract window from buffer
      const windowData = this.streamBuffer.slice(0, windowSamples);

      if (this.useWorker && this.workerReady) {
        // Send to worker for classification
        this.worker.postMessage({
          type: 'stream-classify',
          payload: {
            audioData: windowData
          }
        });
      } else {
        // Classify on main thread
        try {
          const result = await this._classifyMainThread(
            windowData,
            this.options.config.audio.sampleRate
          );
          this.options.onResult(result);
        } catch (error) {
          console.error('[ClassificationService] Stream classify error:', error);
        }
      }

      // Shift buffer (overlap)
      const hopSamples = Math.floor(windowSamples / 2);
      this.streamBuffer.copyWithin(0, hopSamples);
      this.streamBufferSize -= hopSamples;
    }
  }

  /**
   * Stop real-time streaming
   */
  stopStream() {
    if (!this.isStreaming) return;

    this.isStreaming = false;
    this.streamBuffer = null;
    this.streamBufferSize = 0;

    if (this.useWorker && this.workerReady) {
      this.worker.postMessage({ type: 'stop-stream' });
    }

    console.log('[ClassificationService] Streaming stopped');
  }

  /**
   * Change the model (requires reinitialization)
   * @param {string} modelName - New model name
   */
  async setModel(modelName) {
    if (modelName === this.options.model) return;

    this.options.model = modelName;
    this.initialized = false;

    if (this.worker) {
      this.worker.postMessage({ type: 'dispose', id: 0 });
      this.workerReady = false;
    }

    this.classifier = null;
    await this.init();
  }

  /**
   * Change the CQT backend
   * @param {string} backend - New backend name
   */
  async setCqtBackend(backend) {
    if (backend === this.options.cqtBackend) return;

    this.options.cqtBackend = backend;
    this.cqtExtractor = null;

    // Reinitialize main thread components if needed
    if (!this.useWorker) {
      this.cqtExtractor = new CQTExtractor(backend);
      await this.cqtExtractor.init(this.options.config.audio.sampleRate);
    }
  }

  /**
   * Get the current model info
   */
  getModelInfo() {
    if (this.classifier) {
      return this.classifier.getModelInfo();
    }
    return null;
  }

  /**
   * Check if service is ready
   */
  isReady() {
    return this.initialized && (this.useWorker ? this.workerReady : true);
  }

  /**
   * Get current mode
   */
  getMode() {
    return this.useWorker ? 'worker' : 'main-thread';
  }

  /**
   * Dispose resources
   */
  dispose() {
    this.stopStream();

    if (this.worker) {
      this.worker.postMessage({ type: 'dispose', id: 0 });
      this.worker.terminate();
      this.worker = null;
    }

    this.classifier = null;
    this.cqtExtractor = null;
    this.initialized = false;
    this.workerReady = false;

    console.log('[ClassificationService] Disposed');
  }
}

// Export singleton for shared use across pages
let sharedInstance = null;

/**
 * Get/create a shared ClassificationService instance
 * Useful when multiple components need the same classifier
 */
export function getSharedClassificationService(options = {}) {
  if (!sharedInstance) {
    sharedInstance = new ClassificationService(options);
  }
  return sharedInstance;
}

/**
 * Dispose the shared instance
 */
export function disposeSharedClassificationService() {
  if (sharedInstance) {
    sharedInstance.dispose();
    sharedInstance = null;
  }
}
