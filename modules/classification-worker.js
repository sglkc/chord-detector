/**
 * Classification Worker
 * Runs chord classification in a separate thread to avoid blocking the UI
 * 
 * Supports:
 * - Batch classification (multiple onsets)
 * - Single-shot classification
 * - Real-time streaming classification
 * 
 * This is a classic worker (not ES module) for better browser compatibility.
 */

// Import TensorFlow.js in the worker
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');

// Worker state
let model = null;
let isGraphModel = false;
let config = null;
let cqtInitialized = false;
let isStreaming = false;

// Model input shape (detected from loaded model)
let modelInputBins = 36;
let modelInputFrames = 200;

// CQT configuration
let currentSampleRate = 48000;
let nBins = 36;
let binsPerOctave = 12;
let hopLength = 512;
let fmin = 130.81; // C3
let kernels = null;
let fftSize = 4096;

// Chord labels (must match config.js)
const CHORD_LABELS = [
  'A', 'Am', 'Adim',
  'A#', 'A#m', 'A#dim',
  'B', 'Bm', 'Bdim',
  'C', 'Cm', 'Cdim',
  'C#', 'C#m', 'C#dim',
  'D', 'Dm', 'Ddim',
  'D#', 'D#m', 'D#dim',
  'E', 'Em', 'Edim',
  'F', 'Fm', 'Fdim',
  'F#', 'F#m', 'F#dim',
  'G', 'Gm', 'Gdim',
  'G#', 'G#m', 'G#dim'
];

// Model to MIREX chord mapping
function modelToMirex(chord) {
  if (!chord || chord === 'N') return 'N';
  const mappings = {
    'A': 'A:maj', 'Am': 'A:min', 'Adim': 'A:dim',
    'A#': 'A#:maj', 'A#m': 'A#:min', 'A#dim': 'A#:dim',
    'B': 'B:maj', 'Bm': 'B:min', 'Bdim': 'B:dim',
    'C': 'C:maj', 'Cm': 'C:min', 'Cdim': 'C:dim',
    'C#': 'C#:maj', 'C#m': 'C#:min', 'C#dim': 'C#:dim',
    'D': 'D:maj', 'Dm': 'D:min', 'Ddim': 'D:dim',
    'D#': 'D#:maj', 'D#m': 'D#:min', 'D#dim': 'D#:dim',
    'E': 'E:maj', 'Em': 'E:min', 'Edim': 'E:dim',
    'F': 'F:maj', 'Fm': 'F:min', 'Fdim': 'F:dim',
    'F#': 'F#:maj', 'F#m': 'F#:min', 'F#dim': 'F#:dim',
    'G': 'G:maj', 'Gm': 'G:min', 'Gdim': 'G:dim',
    'G#': 'G#:maj', 'G#m': 'G#:min', 'G#dim': 'G#:dim'
  };
  return mappings[chord] || chord;
}

// ============================================================================
// CQT Implementation (optimized for real-time)
// ============================================================================

/**
 * Initialize CQT kernels for frequency analysis
 */
function initCQT(sr, bins, bpo, hop, minFreq) {
  currentSampleRate = sr;
  nBins = bins;
  binsPerOctave = bpo;
  hopLength = hop;
  fmin = minFreq;

  // Calculate required FFT size based on lowest frequency
  const minWinLen = Math.ceil(sr / fmin * 4);
  fftSize = 1;
  while (fftSize < minWinLen) fftSize <<= 1;
  fftSize = Math.min(fftSize, 8192);

  // Pre-compute frequency bins
  const frequencies = new Float32Array(nBins);
  for (let i = 0; i < nBins; i++) {
    frequencies[i] = fmin * Math.pow(2, i / binsPerOctave);
  }

  // Pre-compute complex kernels for each frequency bin
  kernels = new Array(nBins);
  for (let i = 0; i < nBins; i++) {
    const freq = frequencies[i];
    const winLen = Math.min(Math.ceil(sr / freq * 4), fftSize);
    const halfWin = Math.floor(winLen / 2);

    kernels[i] = {
      freq: freq,
      real: new Float32Array(winLen),
      imag: new Float32Array(winLen),
      window: new Float32Array(winLen),
      length: winLen,
      halfWin: halfWin
    };

    // Hann window
    for (let j = 0; j < winLen; j++) {
      kernels[i].window[j] = 0.5 * (1 - Math.cos(2 * Math.PI * j / (winLen - 1)));
    }

    // Complex exponential (windowed sinusoid)
    const omega = 2 * Math.PI * freq / sr;
    for (let j = 0; j < winLen; j++) {
      const t = j - halfWin;
      kernels[i].real[j] = Math.cos(omega * t) * kernels[i].window[j];
      kernels[i].imag[j] = Math.sin(omega * t) * kernels[i].window[j];
    }
  }

  cqtInitialized = true;
  console.log(`[Worker] CQT initialized: ${nBins} bins, sr=${sr}, fmin=${fmin.toFixed(2)}Hz`);
}

/**
 * Extract CQT features from audio window
 * Uses user-configurable parameters (nBins is set by initCQT)
 * Returns raw CQT with dimensions - caller handles resize for model
 */
function extractCQTFeatures(audioData) {
  if (!cqtInitialized) {
    throw new Error('CQT not initialized');
  }

  const numFrames = Math.max(1, Math.floor((audioData.length - fftSize) / hopLength) + 1);

  // Compute CQT for each frame using configured nBins
  const cqt = new Float32Array(nBins * numFrames);

  for (let frame = 0; frame < numFrames; frame++) {
    const frameStart = frame * hopLength;

    for (let bin = 0; bin < nBins; bin++) {
      const kernel = kernels[bin];
      let realSum = 0;
      let imagSum = 0;

      const start = Math.max(0, frameStart - kernel.halfWin);
      const end = Math.min(audioData.length, frameStart + kernel.length - kernel.halfWin);

      for (let j = start; j < end; j++) {
        const kIdx = j - frameStart + kernel.halfWin;
        if (kIdx >= 0 && kIdx < kernel.length) {
          const sample = audioData[j];
          realSum += sample * kernel.real[kIdx];
          imagSum += sample * kernel.imag[kIdx];
        }
      }

      // Magnitude
      cqt[bin * numFrames + frame] = Math.sqrt(realSum * realSum + imagSum * imagSum);
    }
  }

  // Normalize
  let maxVal = 0;
  for (let i = 0; i < cqt.length; i++) {
    if (cqt[i] > maxVal) maxVal = cqt[i];
  }
  if (maxVal > 0) {
    for (let i = 0; i < cqt.length; i++) {
      cqt[i] /= maxVal;
    }
  }

  return { cqt, numBins: nBins, numFrames };
}

/**
 * Resize CQT features to model's expected input shape
 * Uses bilinear interpolation for better quality
 */
function resizeCQTForModel(cqtData, targetBins, targetFrames) {
  const { cqt, numBins: srcBins, numFrames: srcFrames } = cqtData;

  // If already matching, return as flat array
  if (srcBins === targetBins && srcFrames === targetFrames) {
    return cqt;
  }

  const resized = new Float32Array(targetBins * targetFrames);

  for (let b = 0; b < targetBins; b++) {
    for (let t = 0; t < targetFrames; t++) {
      // Map to source coordinates
      const srcB = (b / targetBins) * srcBins;
      const srcT = (t / targetFrames) * srcFrames;

      // Bilinear interpolation
      const b0 = Math.floor(srcB);
      const b1 = Math.min(b0 + 1, srcBins - 1);
      const t0 = Math.floor(srcT);
      const t1 = Math.min(t0 + 1, srcFrames - 1);

      const bFrac = srcB - b0;
      const tFrac = srcT - t0;

      // Get four neighbors
      const v00 = cqt[b0 * srcFrames + t0];
      const v01 = cqt[b0 * srcFrames + t1];
      const v10 = cqt[b1 * srcFrames + t0];
      const v11 = cqt[b1 * srcFrames + t1];

      // Interpolate
      const v0 = v00 * (1 - tFrac) + v01 * tFrac;
      const v1 = v10 * (1 - tFrac) + v11 * tFrac;
      resized[b * targetFrames + t] = v0 * (1 - bFrac) + v1 * bFrac;
    }
  }

  return resized;
}

// ============================================================================
// Model Loading and Prediction
// ============================================================================

/**
 * Load the TensorFlow.js model
 */
async function loadModel(modelPath) {
  // Try layers model first
  try {
    model = await tf.loadLayersModel(modelPath);
    isGraphModel = false;
    console.log('[Worker] Layers model loaded successfully');

    // Detect model input shape
    if (model.inputs && model.inputs.length > 0 && model.inputs[0].shape) {
      const shape = model.inputs[0].shape;
      if (shape[1]) modelInputBins = shape[1];
      if (shape[2]) modelInputFrames = shape[2];
    }
    console.log(`[Worker] Model expects input: ${modelInputBins}×${modelInputFrames}`);

    // Warm up with detected shape
    const dummyInput = tf.zeros([1, modelInputBins, modelInputFrames, 1]);
    await model.predict(dummyInput).data();
    dummyInput.dispose();

    return { success: true, type: 'layers' };
  } catch (layersError) {
    console.warn('[Worker] Trying graph model...', layersError.message);
  }

  // Fallback to graph model
  try {
    model = await tf.loadGraphModel(modelPath);
    isGraphModel = true;
    console.log('[Worker] Graph model loaded successfully');

    // Detect model input shape
    if (model.inputs && model.inputs.length > 0 && model.inputs[0].shape) {
      const shape = model.inputs[0].shape;
      if (shape[1]) modelInputBins = shape[1];
      if (shape[2]) modelInputFrames = shape[2];
    }
    console.log(`[Worker] Model expects input: ${modelInputBins}×${modelInputFrames}`);

    // Warm up with detected shape
    const dummyInput = tf.zeros([1, modelInputBins, modelInputFrames, 1]);
    const warmupResult = model.execute(dummyInput);
    await warmupResult.data();
    warmupResult.dispose();
    dummyInput.dispose();

    return { success: true, type: 'graph' };
  } catch (graphError) {
    console.error('[Worker] Failed to load model:', graphError);
    return { success: false, error: graphError.message };
  }
}

/**
 * Predict chord from CQT features (already resized to model input shape)
 * Returns full result with top predictions
 */
async function predict(features) {
  if (!model) {
    throw new Error('Model not loaded');
  }

  const inputTensor = tf.tidy(() => {
    let tensor = tf.tensor1d(features);
    tensor = tensor.reshape([modelInputBins, modelInputFrames]);
    tensor = tensor.expandDims(0).expandDims(-1);
    return tensor;
  });

  try {
    const prediction = isGraphModel
      ? model.execute(inputTensor)
      : model.predict(inputTensor);
    const probabilities = await prediction.data();

    // Find top predictions
    const indexed = Array.from(probabilities).map((prob, idx) => ({
      index: idx,
      probability: prob,
      chord: CHORD_LABELS[idx],
      mirexChord: modelToMirex(CHORD_LABELS[idx])
    }));
    indexed.sort((a, b) => b.probability - a.probability);

    const top = indexed[0];
    const topPredictions = indexed.slice(0, 3);

    prediction.dispose();
    inputTensor.dispose();

    return {
      chord: top.chord,
      mirexChord: top.mirexChord,
      confidence: top.probability,
      classIndex: top.index,
      topPredictions: topPredictions
    };
  } catch (error) {
    inputTensor.dispose();
    throw error;
  }
}

/**
 * Classify a single audio segment
 * Extracts CQT with user params, resizes to model input, then predicts
 */
async function classifySingle(audioData) {
  if (!cqtInitialized) {
    throw new Error('CQT not initialized');
  }

  // Extract CQT with user-configured params (nBins, hopLength, etc.)
  const cqtData = extractCQTFeatures(audioData);

  // Resize to model's expected input shape
  const features = resizeCQTForModel(cqtData, modelInputBins, modelInputFrames);

  return await predict(features);
}

/**
 * Process a batch of classification tasks
 */
async function classifyBatch(tasks, audioData, sr, windowSize, audioDuration) {
  const predictions = [];
  const windowSamples = Math.floor(windowSize * sr);

  for (let i = 0; i < tasks.length; i++) {
    const task = tasks[i];
    const startSample = Math.floor(task.time * sr);
    const endSample = Math.min(startSample + windowSamples, audioData.length);

    let endTime;
    if (i < tasks.length - 1) {
      endTime = tasks[i + 1].time;
    } else {
      endTime = audioDuration;
    }

    const windowData = audioData.slice(startSample, endSample);

    if (windowData.length < windowSamples * 0.5) {
      continue;
    }

    try {
      // Extract CQT with user-configured params
      const cqtData = extractCQTFeatures(windowData);

      // Resize to model's expected input shape
      const features = resizeCQTForModel(cqtData, modelInputBins, modelInputFrames);

      const result = await predict(features);

      predictions.push({
        start: task.time,
        end: endTime,
        chord: result.chord,
        confidence: result.confidence,
        mirexChord: result.mirexChord
      });

      // Report progress
      self.postMessage({
        type: 'progress',
        current: i + 1,
        total: tasks.length,
        percent: Math.round(((i + 1) / tasks.length) * 100)
      });
    } catch (error) {
      console.error(`[Worker] Error at onset ${task.time}:`, error);
    }
  }

  return predictions;
}

// ============================================================================
// Message Handler
// ============================================================================

self.onmessage = async function (event) {
  const { type, payload, id } = event.data;

  try {
    switch (type) {
      case 'init': {
        const { modelPath, sampleRate: sr, config: cfg } = payload;
        config = cfg;

        self.postMessage({ type: 'status', message: 'Loading model...' });
        const modelResult = await loadModel(modelPath);

        if (!modelResult.success) {
          self.postMessage({ type: 'error', id, error: modelResult.error });
          return;
        }

        self.postMessage({ type: 'status', message: 'Initializing CQT...' });
        initCQT(
          sr || 48000,
          cfg?.classification?.cqtBins || 36,
          12,
          cfg?.audio?.hopSize || 512,
          cfg?.audio?.minFrequency || 130.81
        );

        self.postMessage({ type: 'ready', id });
        break;
      }

      case 'classify': {
        // Batch classification
        const { onsets, audioData, sampleRate: sr, windowSize, audioDuration } = payload;

        const audioArray = audioData instanceof Float32Array
          ? audioData
          : new Float32Array(audioData);

        const predictions = await classifyBatch(
          onsets,
          audioArray,
          sr,
          windowSize,
          audioDuration
        );

        self.postMessage({ type: 'result', id, predictions });
        break;
      }

      case 'classify-single': {
        // Single shot classification
        const { audioData, sampleRate: sr } = payload;

        const audioArray = audioData instanceof Float32Array
          ? audioData
          : new Float32Array(audioData);

        const prediction = await classifySingle(audioArray);
        self.postMessage({ type: 'result', id, prediction });
        break;
      }

      case 'start-stream': {
        // Start real-time streaming mode
        isStreaming = true;
        console.log('[Worker] Streaming mode started');
        break;
      }

      case 'stream-classify': {
        // Real-time streaming classification
        if (!isStreaming) break;

        const { audioData } = payload;
        const audioArray = audioData instanceof Float32Array
          ? audioData
          : new Float32Array(audioData);

        try {
          const prediction = await classifySingle(audioArray);
          self.postMessage({
            type: 'stream-result',
            prediction,
            timestamp: Date.now()
          });
        } catch (error) {
          console.error('[Worker] Stream classify error:', error);
        }
        break;
      }

      case 'stop-stream': {
        isStreaming = false;
        console.log('[Worker] Streaming mode stopped');
        break;
      }

      case 'dispose': {
        isStreaming = false;
        if (model) {
          model.dispose();
          model = null;
        }
        cqtInitialized = false;
        kernels = null;
        self.postMessage({ type: 'disposed', id });
        break;
      }

      default:
        console.warn('[Worker] Unknown message type:', type);
    }
  } catch (error) {
    console.error('[Worker] Error:', error);
    self.postMessage({ type: 'error', id, error: error.message });
  }
};

console.log('[Worker] Classification worker loaded');
