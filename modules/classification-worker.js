/**
 * Classification Worker
 * Runs chord classification in a separate thread to avoid blocking the UI
 * 
 * This is a classic worker (not ES module) because:
 * 1. importScripts works better with CDN resources
 * 2. Better browser compatibility
 */

// Import TensorFlow.js in the worker
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');

// Worker state
let model = null;
let isGraphModel = false;
let config = null;
let cqtInitialized = false;

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
// Simple CQT Implementation (inline to avoid module imports in worker)
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
  const minWinLen = Math.ceil(sr / fmin * 4); // 4 cycles at minimum frequency
  fftSize = 1;
  while (fftSize < minWinLen) fftSize <<= 1;
  fftSize = Math.min(fftSize, 8192); // Cap for performance

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
 */
function extractCQTFeatures(audioData, targetFrames = 200) {
  if (!cqtInitialized) {
    throw new Error('CQT not initialized');
  }

  const numFrames = Math.floor((audioData.length - fftSize) / hopLength) + 1;
  const actualFrames = Math.max(1, numFrames);

  // Compute CQT for each frame
  const cqt = new Float32Array(nBins * actualFrames);

  for (let frame = 0; frame < actualFrames; frame++) {
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
      const magnitude = Math.sqrt(realSum * realSum + imagSum * imagSum);
      cqt[bin * actualFrames + frame] = magnitude;
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

  // Resize to target frames (200)
  if (actualFrames !== targetFrames) {
    const resized = new Float32Array(nBins * targetFrames);
    for (let bin = 0; bin < nBins; bin++) {
      for (let t = 0; t < targetFrames; t++) {
        const srcT = (t / targetFrames) * actualFrames;
        const srcIdx = Math.min(Math.floor(srcT), actualFrames - 1);
        resized[bin * targetFrames + t] = cqt[bin * actualFrames + srcIdx];
      }
    }
    return resized;
  }

  return cqt;
}

// ============================================================================
// Model Loading and Prediction
// ============================================================================

/**
 * Load the TensorFlow.js model
 */
async function loadModel(modelPath) {
  // Try loading as layers model first
  try {
    model = await tf.loadLayersModel(modelPath);
    isGraphModel = false;
    console.log('[Worker] Layers model loaded successfully');

    // Warm up the model
    const dummyInput = tf.zeros([1, 36, 200, 1]);
    await model.predict(dummyInput).data();
    dummyInput.dispose();

    return { success: true, type: 'layers' };
  } catch (layersError) {
    console.warn('[Worker] Failed to load as layers model, trying graph model...', layersError.message);
  }

  // Fallback to graph model
  try {
    model = await tf.loadGraphModel(modelPath);
    isGraphModel = true;
    console.log('[Worker] Graph model loaded successfully');

    // Warm up the model
    const dummyInput = tf.zeros([1, 36, 200, 1]);
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
 * Predict chord from CQT features
 */
async function predict(features) {
  if (!model) {
    throw new Error('Model not loaded');
  }

  let numBins = 36;
  let numFrames = 200;

  if (model.inputs && model.inputs.length > 0 && model.inputs[0].shape) {
    const inputShape = model.inputs[0].shape;
    if (inputShape[1]) numBins = inputShape[1];
    if (inputShape[2]) numFrames = inputShape[2];
  }

  const inputTensor = tf.tidy(() => {
    let tensor = tf.tensor1d(features);
    tensor = tensor.reshape([numBins, numFrames]);
    tensor = tensor.expandDims(0).expandDims(-1);
    return tensor;
  });

  try {
    const prediction = isGraphModel
      ? model.execute(inputTensor)
      : model.predict(inputTensor);
    const probabilities = await prediction.data();

    let maxProb = 0;
    let maxIndex = 0;

    for (let i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }

    const chord = CHORD_LABELS[maxIndex];
    const mirexChord = modelToMirex(chord);

    prediction.dispose();
    inputTensor.dispose();

    return {
      chord: chord,
      mirexChord: mirexChord,
      confidence: maxProb,
      classIndex: maxIndex
    };
  } catch (error) {
    inputTensor.dispose();
    throw error;
  }
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

    // Determine end time
    let endTime;
    if (i < tasks.length - 1) {
      endTime = tasks[i + 1].time;
    } else {
      endTime = audioDuration;
    }

    // Extract window
    const windowData = audioData.slice(startSample, endSample);

    if (windowData.length < windowSamples * 0.5) {
      continue; // Skip if window too short
    }

    try {
      // Extract CQT features
      const cqtFeatures = extractCQTFeatures(windowData, 200);

      // Classify
      const result = await predict(cqtFeatures);

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
      console.error(`[Worker] Error classifying onset at ${task.time}:`, error);
    }
  }

  return predictions;
}

// ============================================================================
// Message Handler
// ============================================================================

/**
 * Handle messages from the main thread
 */
self.onmessage = async function (event) {
  const { type, payload, id } = event.data;

  try {
    switch (type) {
      case 'init': {
        // Initialize model and CQT
        const { modelPath, sampleRate: sr, config: cfg } = payload;
        config = cfg;

        self.postMessage({ type: 'status', message: 'Loading model...' });
        const modelResult = await loadModel(modelPath);

        if (!modelResult.success) {
          self.postMessage({ type: 'error', id, error: modelResult.error });
          return;
        }

        // Initialize CQT
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
        const { onsets, audioData, sampleRate: sr, windowSize, audioDuration } = payload;

        // Convert audioData back to Float32Array if needed
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

        self.postMessage({
          type: 'result',
          id,
          predictions
        });
        break;
      }

      case 'dispose': {
        // Clean up resources
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
    self.postMessage({
      type: 'error',
      id,
      error: error.message
    });
  }
};

console.log('[Worker] Classification worker loaded');
