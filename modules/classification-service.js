/**
 * Classification Service
 * 
 * A modular, reusable service for chord classification that works across pages.
 * All heavy processing happens in a Web Worker (non-blocking).
 * 
 * Modes:
 * - Full pipeline: CQT extraction + onset detection + classification (processAudio)
 * - Single-shot: Classify a single audio segment (classify)
 * - Real-time streaming: Continuous classification from microphone
 * 
 * Usage:
 *   const service = new ClassificationService({
 *       model: 'graph-v2',
 *       onProgress: (percent, message, stage) => updateUI(percent, message)
 *   });
 *   
 *   await service.init();
 *   const { fullCQT, onsets, predictions } = await service.processAudio(audioData, duration);
 */

import { CONFIG } from './config.js';

export const ClassificationMode = {
  WORKER: 'worker',
  AUTO: 'auto'
};

export class ClassificationService {
  constructor(options = {}) {
    this.options = {
      model: options.model || CONFIG.classification.model || 'graph-v2',
      mode: options.mode || ClassificationMode.AUTO,
      onProgress: options.onProgress || (() => { }),
      onResult: options.onResult || (() => { }),
      config: options.config || CONFIG
    };

    this.initialized = false;
    this.worker = null;
    this.workerReady = false;

    this.pendingCallbacks = new Map();
    this.messageId = 0;

    this.isStreaming = false;
    this.streamBuffer = null;
    this.streamBufferSize = 0;
  }

  async init() {
    if (this.initialized) {
      console.log('[ClassificationService] Already initialized');
      return;
    }

    try {
      await this._initWorker();
      console.log('[ClassificationService] Worker initialized');
    } catch (error) {
      console.error('[ClassificationService] Worker init failed:', error);
      throw error;
    }

    this.initialized = true;
    console.log('[ClassificationService] Ready');
  }

  /**
   * Serialize config for sending to worker (removes functions)
   */
  _serializeConfig(config) {
    return {
      audio: {
        sampleRate: config.audio?.sampleRate || 48000,
        hopSize: config.audio?.hopSize || 512,
        minFrequency: config.audio?.minFrequency || 130.81
      },
      onset: {
        threshold: config.onset?.threshold || 0.15,
        minInterval: config.onset?.minInterval || 100,
        preBuffer: config.onset?.preBuffer || 50,
        frameSize: config.onset?.frameSize || 2048,
        smoothingWindow: config.onset?.smoothingWindow || 5,
        ignoreSubsequentOnsets: config.onset?.ignoreSubsequentOnsets || false
      },
      classification: {
        model: config.classification?.model || 'graph-v2',
        windowSize: config.classification?.windowSize || 2.0,
        cqtBins: config.classification?.cqtBins || 36,
        cqtTimeFrames: config.classification?.cqtTimeFrames || 200,
        confidenceThreshold: config.classification?.confidenceThreshold || 0.5
      }
    };
  }

  async _initWorker() {
    return new Promise((resolve, reject) => {
      try {
        this.worker = new Worker('./modules/classification-worker.js');

        this.worker.onmessage = (event) => this._handleWorkerMessage(event.data);
        this.worker.onerror = (error) => {
          console.error('[ClassificationService] Worker error:', error);
          reject(error);
        };

        const id = ++this.messageId;
        // Create absolute URL that works in worker context
        const baseUrl = new URL('./', window.location.href).href;
        const modelPath = `${baseUrl}models/${this.options.model}/model.json`;

        this.pendingCallbacks.set(id, { resolve, reject });

        this.worker.postMessage({
          type: 'init',
          id,
          payload: {
            modelPath,
            sampleRate: this.options.config.audio.sampleRate,
            config: this._serializeConfig(this.options.config)
          }
        });

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
        this.options.onProgress(rest.percent, rest.message, rest.stage);
        break;

      case 'result':
        if (this.pendingCallbacks.has(id)) {
          this.pendingCallbacks.get(id).resolve(rest);
          this.pendingCallbacks.delete(id);
        }
        break;

      case 'stream-result':
        this.options.onResult(rest.prediction);
        break;

      case 'stream-ready':
        console.log('[ClassificationService] Stream ready');
        break;

      case 'onset-detected':
        console.log('[ClassificationService] Onset detected at', rest.timestamp);
        break;

      case 'error':
        console.error('[ClassificationService] Worker error:', rest.error);
        if (this.pendingCallbacks.has(id)) {
          this.pendingCallbacks.get(id).reject(new Error(rest.error));
          this.pendingCallbacks.delete(id);
        }
        break;
    }
  }

  _sendWorkerMessage(type, payload) {
    return new Promise((resolve, reject) => {
      const id = ++this.messageId;
      this.pendingCallbacks.set(id, { resolve, reject });
      this.worker.postMessage({ type, id, payload });
    });
  }

  /**
   * Process audio through full pipeline (CQT + onset detection + classification)
   * @param {Float32Array} audioData - Audio samples
   * @param {number} audioDuration - Audio duration in seconds
   * @returns {Object} { fullCQT, onsets, predictions }
   */
  async processAudio(audioData, audioDuration) {
    if (!this.initialized) await this.init();

    if (!this.workerReady) {
      throw new Error('Worker not ready');
    }

    const result = await this._sendWorkerMessage('process-audio', {
      audioData: audioData,
      config: this._serializeConfig(this.options.config),
      audioDuration
    });

    return {
      fullCQT: result.fullCQT,
      onsets: result.onsets,
      predictions: result.predictions
    };
  }

  /**
   * Classify a single audio segment
   * @param {Float32Array} audioData - Audio samples
   * @returns {Object} Classification result
   */
  async classify(audioData) {
    if (!this.initialized) await this.init();

    if (!this.workerReady) {
      throw new Error('Worker not ready');
    }

    const result = await this._sendWorkerMessage('classify-single', {
      audioData: audioData
    });

    return result.prediction;
  }

  /**
   * Start real-time streaming classification
   */
  startStream(onResult = null, options = {}) {
    if (this.isStreaming) {
      console.warn('[ClassificationService] Already streaming');
      return;
    }

    this.isStreaming = true;
    this.streamBuffer = new Float32Array(options.bufferSize || 96000);
    this.streamBufferSize = 0;

    if (onResult) {
      this.options.onResult = onResult;
    }

    if (this.workerReady) {
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
   */
  async feedAudio(samples) {
    if (!this.isStreaming) return;

    // Append to buffer
    if (this.streamBufferSize + samples.length > this.streamBuffer.length) {
      const shift = samples.length;
      this.streamBuffer.copyWithin(0, shift);
      this.streamBufferSize = Math.max(0, this.streamBufferSize - shift);
    }

    this.streamBuffer.set(samples, this.streamBufferSize);
    this.streamBufferSize += samples.length;

    const windowSamples = Math.floor(
      this.options.config.classification.windowSize *
      this.options.config.audio.sampleRate
    );

    if (this.streamBufferSize >= windowSamples) {
      const windowData = this.streamBuffer.slice(0, windowSamples);

      if (this.workerReady) {
        this.worker.postMessage({
          type: 'stream-classify',
          payload: { audioData: windowData }
        });
      }

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

    if (this.workerReady) {
      this.worker.postMessage({ type: 'stop-stream' });
    }

    console.log('[ClassificationService] Streaming stopped');
  }

  /**
   * Change the model
   */
  async setModel(modelName) {
    if (modelName === this.options.model) return;

    this.options.model = modelName;
    this.initialized = false;
    this.workerReady = false;

    if (this.worker) {
      this.worker.postMessage({ type: 'dispose', id: 0 });
    }

    await this.init();
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig) {
    this.options.config = { ...this.options.config, ...newConfig };
  }

  isReady() {
    return this.initialized && this.workerReady;
  }

  dispose() {
    this.stopStream();

    if (this.worker) {
      this.worker.postMessage({ type: 'dispose', id: 0 });
      this.worker.terminate();
      this.worker = null;
    }

    this.initialized = false;
    this.workerReady = false;

    console.log('[ClassificationService] Disposed');
  }
}

// Singleton for shared use
let sharedInstance = null;

export function getSharedClassificationService(options = {}) {
  if (!sharedInstance) {
    sharedInstance = new ClassificationService(options);
  }
  return sharedInstance;
}

export function disposeSharedClassificationService() {
  if (sharedInstance) {
    sharedInstance.dispose();
    sharedInstance = null;
  }
}
