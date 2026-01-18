/**
 * Unified Classification Worker
 * 
 * Handles ALL audio processing in a separate thread:
 * - Full CQT extraction (for visualization)
 * - Onset detection (spectral flux)
 * - Chord classification
 * 
 * Message Types:
 * - 'init': Load model, initialize CQT
 * - 'process-audio': Full pipeline (CQT → onsets → classify) with progress
 * - 'classify-single': Single audio segment classification
 * - 'stream-classify': Real-time classification (minimal overhead)
 * - 'dispose': Clean up resources
 * 
 * All parameters are configurable from UI.
 */

// Import TensorFlow.js
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');

// ============================================================================
// Worker State
// ============================================================================

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
let fmin = 130.81;
let kernels = null;
let fftSize = 4096;

// Chord labels
const CHORD_LABELS = [
  'A', 'Am', 'Adim', 'A#', 'A#m', 'A#dim',
  'B', 'Bm', 'Bdim', 'C', 'Cm', 'Cdim',
  'C#', 'C#m', 'C#dim', 'D', 'Dm', 'Ddim',
  'D#', 'D#m', 'D#dim', 'E', 'Em', 'Edim',
  'F', 'Fm', 'Fdim', 'F#', 'F#m', 'F#dim',
  'G', 'Gm', 'Gdim', 'G#', 'G#m', 'G#dim'
];

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
// Progress Reporting
// ============================================================================

function reportProgress(stage, percent, message, data = null) {
  self.postMessage({
    type: 'progress',
    stage,
    percent,
    message,
    data
  });
}

// ============================================================================
// CQT Implementation
// ============================================================================

function initCQT(sr, bins, bpo, hop, minFreq) {
  currentSampleRate = sr;
  nBins = bins;
  binsPerOctave = bpo;
  hopLength = hop;
  fmin = minFreq;

  const minWinLen = Math.ceil(sr / fmin * 4);
  fftSize = 1;
  while (fftSize < minWinLen) fftSize <<= 1;
  fftSize = Math.min(fftSize, 8192);

  const frequencies = new Float32Array(nBins);
  for (let i = 0; i < nBins; i++) {
    frequencies[i] = fmin * Math.pow(2, i / binsPerOctave);
  }

  kernels = new Array(nBins);
  for (let i = 0; i < nBins; i++) {
    const freq = frequencies[i];
    const winLen = Math.min(Math.ceil(sr / freq * 4), fftSize);
    const halfWin = Math.floor(winLen / 2);

    kernels[i] = {
      freq, real: new Float32Array(winLen), imag: new Float32Array(winLen),
      window: new Float32Array(winLen), length: winLen, halfWin
    };

    for (let j = 0; j < winLen; j++) {
      kernels[i].window[j] = 0.5 * (1 - Math.cos(2 * Math.PI * j / (winLen - 1)));
    }

    const omega = 2 * Math.PI * freq / sr;
    for (let j = 0; j < winLen; j++) {
      const t = j - halfWin;
      kernels[i].real[j] = Math.cos(omega * t) * kernels[i].window[j];
      kernels[i].imag[j] = Math.sin(omega * t) * kernels[i].window[j];
    }
  }

  cqtInitialized = true;
  console.log(`[Worker] CQT initialized: ${nBins} bins, sr=${sr}, hop=${hop}, fmin=${fmin.toFixed(2)}Hz`);
}

/**
 * Extract full CQT spectrogram for visualization
 * Returns 2D array [numFrames][numBins] for easy rendering
 */
function extractFullCQT(audioData) {
  if (!cqtInitialized) throw new Error('CQT not initialized');

  const numFrames = Math.max(1, Math.floor((audioData.length - fftSize) / hopLength) + 1);
  const magnitudes = [];

  for (let frame = 0; frame < numFrames; frame++) {
    const frameStart = frame * hopLength;
    const frameMags = new Float32Array(nBins);

    for (let bin = 0; bin < nBins; bin++) {
      const kernel = kernels[bin];
      let realSum = 0, imagSum = 0;

      const start = Math.max(0, frameStart - kernel.halfWin);
      const end = Math.min(audioData.length, frameStart + kernel.length - kernel.halfWin);

      for (let j = start; j < end; j++) {
        const kIdx = j - frameStart + kernel.halfWin;
        if (kIdx >= 0 && kIdx < kernel.length) {
          realSum += audioData[j] * kernel.real[kIdx];
          imagSum += audioData[j] * kernel.imag[kIdx];
        }
      }

      frameMags[bin] = Math.sqrt(realSum * realSum + imagSum * imagSum);
    }

    magnitudes.push(frameMags);
  }

  // Generate time array
  const times = new Float32Array(numFrames);
  for (let i = 0; i < numFrames; i++) {
    times[i] = (i * hopLength) / currentSampleRate;
  }

  return { magnitudes, times, numFrames, numBins: nBins, hopSize: hopLength };
}

/**
 * Extract CQT features for classification (returns flat array with dimensions)
 */
function extractCQTFeatures(audioData) {
  if (!cqtInitialized) throw new Error('CQT not initialized');

  const numFrames = Math.max(1, Math.floor((audioData.length - fftSize) / hopLength) + 1);
  const cqt = new Float32Array(nBins * numFrames);

  for (let frame = 0; frame < numFrames; frame++) {
    const frameStart = frame * hopLength;

    for (let bin = 0; bin < nBins; bin++) {
      const kernel = kernels[bin];
      let realSum = 0, imagSum = 0;

      const start = Math.max(0, frameStart - kernel.halfWin);
      const end = Math.min(audioData.length, frameStart + kernel.length - kernel.halfWin);

      for (let j = start; j < end; j++) {
        const kIdx = j - frameStart + kernel.halfWin;
        if (kIdx >= 0 && kIdx < kernel.length) {
          realSum += audioData[j] * kernel.real[kIdx];
          imagSum += audioData[j] * kernel.imag[kIdx];
        }
      }

      cqt[bin * numFrames + frame] = Math.sqrt(realSum * realSum + imagSum * imagSum);
    }
  }

  // Normalize
  let maxVal = 0;
  for (let i = 0; i < cqt.length; i++) if (cqt[i] > maxVal) maxVal = cqt[i];
  if (maxVal > 0) for (let i = 0; i < cqt.length; i++) cqt[i] /= maxVal;

  return { cqt, numBins: nBins, numFrames };
}

/**
 * Resize CQT to model input shape using bilinear interpolation
 */
function resizeCQTForModel(cqtData, targetBins, targetFrames) {
  const { cqt, numBins: srcBins, numFrames: srcFrames } = cqtData;

  if (srcBins === targetBins && srcFrames === targetFrames) return cqt;

  const resized = new Float32Array(targetBins * targetFrames);

  for (let b = 0; b < targetBins; b++) {
    for (let t = 0; t < targetFrames; t++) {
      const srcB = (b / targetBins) * srcBins;
      const srcT = (t / targetFrames) * srcFrames;

      const b0 = Math.floor(srcB), b1 = Math.min(b0 + 1, srcBins - 1);
      const t0 = Math.floor(srcT), t1 = Math.min(t0 + 1, srcFrames - 1);
      const bFrac = srcB - b0, tFrac = srcT - t0;

      const v00 = cqt[b0 * srcFrames + t0], v01 = cqt[b0 * srcFrames + t1];
      const v10 = cqt[b1 * srcFrames + t0], v11 = cqt[b1 * srcFrames + t1];

      const v0 = v00 * (1 - tFrac) + v01 * tFrac;
      const v1 = v10 * (1 - tFrac) + v11 * tFrac;
      resized[b * targetFrames + t] = v0 * (1 - bFrac) + v1 * bFrac;
    }
  }

  return resized;
}

// ============================================================================
// Onset Detection (Spectral Flux)
// ============================================================================

function detectOnsets(audioData, cfg) {
  const sr = cfg.audio?.sampleRate || currentSampleRate;
  const hop = cfg.audio?.hopSize || hopLength;
  const frameSize = cfg.onset?.frameSize || 2048;
  const threshold = cfg.onset?.threshold || 0.15;
  const minIntervalMs = cfg.onset?.minInterval || 100;
  const minIntervalSamples = (minIntervalMs / 1000) * sr;
  const smoothingWindow = cfg.onset?.smoothingWindow || 5;

  // Calculate spectral flux
  const numFrames = Math.floor((audioData.length - frameSize) / hop) + 1;
  const flux = new Float32Array(numFrames);
  let prevSpectrum = null;

  for (let i = 0; i < numFrames; i++) {
    const startSample = i * hop;
    const frame = new Float32Array(frameSize);
    for (let j = 0; j < frameSize && startSample + j < audioData.length; j++) {
      frame[j] = audioData[startSample + j];
    }

    // Hanning window
    for (let j = 0; j < frameSize; j++) {
      frame[j] *= 0.5 * (1 - Math.cos(2 * Math.PI * j / (frameSize - 1)));
    }

    // Magnitude spectrum (simplified DFT for lower frequencies)
    const numBinsSpec = Math.min(frameSize / 2, 256);
    const spectrum = new Float32Array(numBinsSpec);

    for (let k = 0; k < numBinsSpec; k++) {
      let real = 0, imag = 0;
      for (let n = 0; n < frameSize; n++) {
        const angle = (2 * Math.PI * k * n) / frameSize;
        real += frame[n] * Math.cos(angle);
        imag -= frame[n] * Math.sin(angle);
      }
      spectrum[k] = Math.sqrt(real * real + imag * imag) / frameSize;
    }

    if (prevSpectrum) {
      let fluxValue = 0;
      for (let j = 0; j < spectrum.length; j++) {
        const diff = spectrum[j] - prevSpectrum[j];
        if (diff > 0) fluxValue += diff * diff;
      }
      flux[i] = Math.sqrt(fluxValue);
    }

    prevSpectrum = spectrum;
  }

  // Smooth flux
  const smoothed = new Float32Array(flux.length);
  const halfWin = Math.floor(smoothingWindow / 2);
  for (let i = 0; i < flux.length; i++) {
    let sum = 0, count = 0;
    for (let j = -halfWin; j <= halfWin; j++) {
      const idx = i + j;
      if (idx >= 0 && idx < flux.length) { sum += flux[idx]; count++; }
    }
    smoothed[i] = sum / count;
  }

  // Normalize
  let maxFlux = 0;
  for (let i = 0; i < smoothed.length; i++) if (smoothed[i] > maxFlux) maxFlux = smoothed[i];
  if (maxFlux > 0) for (let i = 0; i < smoothed.length; i++) smoothed[i] /= maxFlux;

  // Adaptive threshold
  const adaptiveThreshold = new Float32Array(smoothed.length);
  for (let i = 0; i < smoothed.length; i++) {
    const start = Math.max(0, i - 10);
    const end = Math.min(smoothed.length, i + 11);
    const local = [];
    for (let j = start; j < end; j++) local.push(smoothed[j]);
    local.sort((a, b) => a - b);
    const median = local[Math.floor(local.length / 2)];
    adaptiveThreshold[i] = threshold + 0.5 * median;
  }

  // Peak picking
  const peaks = [];
  const minDist = minIntervalSamples / hop;
  let lastPeakIdx = -minDist;

  for (let i = 1; i < smoothed.length - 1; i++) {
    if (smoothed[i] > smoothed[i - 1] && smoothed[i] > smoothed[i + 1] && smoothed[i] > adaptiveThreshold[i]) {
      if (i - lastPeakIdx >= minDist) {
        peaks.push({ index: i, value: smoothed[i] });
        lastPeakIdx = i;
      } else if (peaks.length > 0 && smoothed[i] > peaks[peaks.length - 1].value) {
        peaks[peaks.length - 1] = { index: i, value: smoothed[i] };
        lastPeakIdx = i;
      }
    }
  }

  // Convert to timestamps
  return peaks.map(p => ({
    time: (p.index * hop) / sr,
    strength: p.value,
    sample: p.index * hop
  }));
}

/**
 * Filter subsequent onsets within window duration
 */
function filterSubsequentOnsets(onsets, windowSize) {
  if (onsets.length === 0) return onsets;

  const filtered = [];
  let lastKeptTime = -Infinity;

  for (const onset of onsets) {
    if (onset.time >= lastKeptTime + windowSize) {
      filtered.push(onset);
      lastKeptTime = onset.time;
    }
  }

  return filtered;
}

// ============================================================================
// Model Loading and Prediction
// ============================================================================

async function loadModel(modelPath) {
  try {
    model = await tf.loadLayersModel(modelPath);
    isGraphModel = false;
    console.log('[Worker] Layers model loaded');

    if (model.inputs?.[0]?.shape) {
      const shape = model.inputs[0].shape;
      if (shape[1]) modelInputBins = shape[1];
      if (shape[2]) modelInputFrames = shape[2];
    }
    console.log(`[Worker] Model expects: ${modelInputBins}×${modelInputFrames}`);

    const dummyInput = tf.zeros([1, modelInputBins, modelInputFrames, 1]);
    await model.predict(dummyInput).data();
    dummyInput.dispose();

    return { success: true, type: 'layers' };
  } catch (e) {
    console.warn('[Worker] Trying graph model...', e.message);
  }

  try {
    model = await tf.loadGraphModel(modelPath);
    isGraphModel = true;
    console.log('[Worker] Graph model loaded');

    if (model.inputs?.[0]?.shape) {
      const shape = model.inputs[0].shape;
      if (shape[1]) modelInputBins = shape[1];
      if (shape[2]) modelInputFrames = shape[2];
    }
    console.log(`[Worker] Model expects: ${modelInputBins}×${modelInputFrames}`);

    const dummyInput = tf.zeros([1, modelInputBins, modelInputFrames, 1]);
    const warmup = model.execute(dummyInput);
    await warmup.data();
    warmup.dispose();
    dummyInput.dispose();

    return { success: true, type: 'graph' };
  } catch (e) {
    console.error('[Worker] Failed to load model:', e);
    return { success: false, error: e.message };
  }
}

async function predict(features) {
  if (!model) throw new Error('Model not loaded');

  const inputTensor = tf.tidy(() => {
    let t = tf.tensor1d(features);
    t = t.reshape([modelInputBins, modelInputFrames]);
    t = t.expandDims(0).expandDims(-1);
    return t;
  });

  try {
    const pred = isGraphModel ? model.execute(inputTensor) : model.predict(inputTensor);
    const probs = await pred.data();

    const indexed = Array.from(probs).map((p, i) => ({
      index: i, probability: p, chord: CHORD_LABELS[i], mirexChord: modelToMirex(CHORD_LABELS[i])
    }));
    indexed.sort((a, b) => b.probability - a.probability);

    pred.dispose();
    inputTensor.dispose();

    return {
      chord: indexed[0].chord,
      mirexChord: indexed[0].mirexChord,
      confidence: indexed[0].probability,
      classIndex: indexed[0].index,
      topPredictions: indexed.slice(0, 3)
    };
  } catch (e) {
    inputTensor.dispose();
    throw e;
  }
}

async function classifySingle(audioData) {
  if (!cqtInitialized) throw new Error('CQT not initialized');
  const cqtData = extractCQTFeatures(audioData);
  const features = resizeCQTForModel(cqtData, modelInputBins, modelInputFrames);
  return await predict(features);
}

// ============================================================================
// Full Processing Pipeline
// ============================================================================

async function processAudio(audioData, cfg, audioDuration) {
  const results = { fullCQT: null, onsets: [], predictions: [] };

  // Step 1: Extract full CQT for visualization
  reportProgress('cqt', 10, 'Extracting CQT spectrogram...');
  results.fullCQT = extractFullCQT(audioData);
  reportProgress('cqt', 30, `CQT extracted: ${results.fullCQT.numFrames} frames`);

  // Step 2: Detect onsets
  reportProgress('onset', 35, 'Detecting onsets...');
  let onsets = detectOnsets(audioData, cfg);
  reportProgress('onset', 45, `Found ${onsets.length} onsets`);

  // Step 2.5: Filter subsequent onsets if enabled
  if (cfg.onset?.ignoreSubsequentOnsets) {
    onsets = filterSubsequentOnsets(onsets, cfg.classification?.windowSize || 2.0);
    reportProgress('onset', 50, `Filtered to ${onsets.length} onsets`);
  }

  results.onsets = onsets;

  // Step 3: Classify each onset
  if (onsets.length === 0) {
    reportProgress('classify', 100, 'No onsets to classify');
    return results;
  }

  const sr = cfg.audio?.sampleRate || currentSampleRate;
  const windowSize = cfg.classification?.windowSize || 2.0;
  const windowSamples = Math.floor(windowSize * sr);

  for (let i = 0; i < onsets.length; i++) {
    const onset = onsets[i];
    const startSample = Math.floor(onset.time * sr);
    const endSample = Math.min(startSample + windowSamples, audioData.length);

    const endTime = (i < onsets.length - 1) ? onsets[i + 1].time : audioDuration;

    const windowData = audioData.slice(startSample, endSample);
    if (windowData.length < windowSamples * 0.5) continue;

    try {
      const cqtData = extractCQTFeatures(windowData);
      const features = resizeCQTForModel(cqtData, modelInputBins, modelInputFrames);
      const result = await predict(features);

      results.predictions.push({
        start: onset.time,
        end: endTime,
        chord: result.chord,
        confidence: result.confidence,
        mirexChord: result.mirexChord
      });

      const percent = 50 + Math.round(((i + 1) / onsets.length) * 50);
      reportProgress('classify', percent, `Classifying onset ${i + 1}/${onsets.length}: ${result.mirexChord}`);
    } catch (e) {
      console.error(`[Worker] Error at onset ${onset.time}:`, e);
    }
  }

  reportProgress('complete', 100, 'Processing complete');
  return results;
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

        reportProgress('init', 5, 'Loading model...');
        const result = await loadModel(modelPath);
        if (!result.success) {
          self.postMessage({ type: 'error', id, error: result.error });
          return;
        }

        reportProgress('init', 50, 'Initializing CQT...');
        initCQT(
          sr || 48000,
          cfg?.classification?.cqtBins || 36,
          12,
          cfg?.audio?.hopSize || 512,
          cfg?.audio?.minFrequency || 130.81
        );

        reportProgress('init', 100, 'Ready');
        self.postMessage({ type: 'ready', id });
        break;
      }

      case 'process-audio': {
        const { audioData, config: cfg, audioDuration } = payload;
        const audioArray = audioData instanceof Float32Array ? audioData : new Float32Array(audioData);

        // Reinitialize CQT if config changed
        const newBins = cfg?.classification?.cqtBins || 36;
        const newHop = cfg?.audio?.hopSize || 512;
        const newFmin = cfg?.audio?.minFrequency || 130.81;
        const newSr = cfg?.audio?.sampleRate || 48000;

        if (nBins !== newBins || hopLength !== newHop || fmin !== newFmin || currentSampleRate !== newSr) {
          initCQT(newSr, newBins, 12, newHop, newFmin);
        }

        const results = await processAudio(audioArray, cfg, audioDuration);
        self.postMessage({ type: 'result', id, ...results });
        break;
      }

      case 'classify-single': {
        const { audioData } = payload;
        const audioArray = audioData instanceof Float32Array ? audioData : new Float32Array(audioData);
        const prediction = await classifySingle(audioArray);
        self.postMessage({ type: 'result', id, prediction });
        break;
      }

      case 'start-stream': {
        isStreaming = true;
        console.log('[Worker] Streaming started');
        break;
      }

      case 'stream-classify': {
        if (!isStreaming) break;
        const { audioData } = payload;
        const audioArray = audioData instanceof Float32Array ? audioData : new Float32Array(audioData);
        try {
          const prediction = await classifySingle(audioArray);
          self.postMessage({ type: 'stream-result', prediction, timestamp: Date.now() });
        } catch (e) {
          console.error('[Worker] Stream error:', e);
        }
        break;
      }

      case 'stop-stream': {
        isStreaming = false;
        console.log('[Worker] Streaming stopped');
        break;
      }

      case 'dispose': {
        isStreaming = false;
        if (model) { model.dispose(); model = null; }
        cqtInitialized = false;
        kernels = null;
        self.postMessage({ type: 'disposed', id });
        break;
      }

      default:
        console.warn('[Worker] Unknown message:', type);
    }
  } catch (e) {
    console.error('[Worker] Error:', e);
    self.postMessage({ type: 'error', id, error: e.message });
  }
};

console.log('[Worker] Unified classification worker loaded');
