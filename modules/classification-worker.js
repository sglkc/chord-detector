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

// Real-time streaming state
let streamConfig = null;
let streamBuffer = null;
let streamBufferSize = 0;
let lastOnsetTime = -Infinity;
let isCapturingWindow = false;
let windowBuffer = null;
let windowBufferSize = 0;
let windowStartTime = 0;

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
  'A#dim', 'A#', 'A#m', 'Adim', 'A', 'Am',
  'Bdim', 'B', 'Bm', 'C#dim', 'C#', 'C#m',
  'Cdim', 'C', 'Cm', 'D#dim', 'D#', 'D#m',
  'Ddim', 'D', 'Dm', 'Edim', 'E', 'Em',
  'F#dim', 'F#', 'F#m', 'Fdim', 'F', 'Fm',
  'G#dim', 'G#', 'G#m', 'Gdim', 'G', 'Gm'
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
// CQT Implementation (librosa-compatible)
// Matches librosa.cqt() output for accurate classification
// ============================================================================

// Complex number class for FFT
class Complex {
  constructor(real = 0, imag = 0) {
    this.real = real;
    this.imag = imag;
  }
  add(other) { return new Complex(this.real + other.real, this.imag + other.imag); }
  sub(other) { return new Complex(this.real - other.real, this.imag - other.imag); }
  mul(other) {
    return new Complex(
      this.real * other.real - this.imag * other.imag,
      this.real * other.imag + this.imag * other.real
    );
  }
  conj() { return new Complex(this.real, -this.imag); }
  magnitude() { return Math.sqrt(this.real * this.real + this.imag * this.imag); }
  static fromPolar(r, theta) { return new Complex(r * Math.cos(theta), r * Math.sin(theta)); }
}

// FFT implementation (Cooley-Tukey radix-2)
function bitReverse(x, bits) {
  let result = 0;
  for (let i = 0; i < bits; i++) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}

function fft(signal) {
  const n = signal.length;
  const bits = Math.log2(n);

  // Bit-reversal permutation
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    result[i] = new Complex(signal[j], 0);
  }

  // Cooley-Tukey iterative FFT
  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const tableStep = n / size;

    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const theta = -2 * Math.PI * j * tableStep / n;
        const twiddle = Complex.fromPolar(1, theta);

        const a = result[i + j];
        const b = result[i + j + halfSize].mul(twiddle);

        result[i + j] = a.add(b);
        result[i + j + halfSize] = a.sub(b);
      }
    }
  }

  return result;
}

function fftComplex(signal) {
  const n = signal.length;
  const bits = Math.log2(n);

  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    result[i] = new Complex(signal[j].real, signal[j].imag);
  }

  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const tableStep = n / size;

    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const theta = -2 * Math.PI * j * tableStep / n;
        const twiddle = Complex.fromPolar(1, theta);

        const a = result[i + j];
        const b = result[i + j + halfSize].mul(twiddle);

        result[i + j] = a.add(b);
        result[i + j + halfSize] = a.sub(b);
      }
    }
  }

  return result;
}

function nextPowerOf2(n) {
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

// CQT state
let cqtQ = 0;
let cqtFrequencies = null;
let cqtKernelLengths = null;
let cqtKernelsFFT = null;
let cqtFFTLength = 0;

function initCQT(sr, bins, bpo, hop, minFreq) {
  currentSampleRate = sr;
  nBins = bins;
  binsPerOctave = bpo;
  hopLength = hop;
  fmin = minFreq;

  // Calculate Q factor (constant for all bins)
  // Q = 1 / (2^(1/bins_per_octave) - 1) ≈ 16.82 for 12 bins per octave
  cqtQ = 1 / (Math.pow(2, 1 / binsPerOctave) - 1);

  // Calculate center frequencies for each bin
  cqtFrequencies = new Float32Array(nBins);
  for (let k = 0; k < nBins; k++) {
    cqtFrequencies[k] = fmin * Math.pow(2, k / binsPerOctave);
  }

  // Calculate window lengths for each bin
  cqtKernelLengths = new Int32Array(nBins);
  for (let k = 0; k < nBins; k++) {
    cqtKernelLengths[k] = Math.ceil(cqtQ * sr / cqtFrequencies[k]);
  }

  // Find maximum kernel length and determine FFT size
  const maxKernelLength = Math.max(...cqtKernelLengths);
  cqtFFTLength = nextPowerOf2(maxKernelLength);
  fftSize = cqtFFTLength;

  // Pre-compute CQT kernels (stored as conjugate of their FFT)
  cqtKernelsFFT = computeCQTKernels();

  cqtInitialized = true;
  console.log(`[Worker] CQT initialized (librosa-compatible):`);
  console.log(`  Q factor: ${cqtQ.toFixed(4)}, FFT length: ${cqtFFTLength}`);
  console.log(`  ${nBins} bins, sr=${sr}, hop=${hop}, fmin=${fmin.toFixed(2)}Hz`);
}

function computeCQTKernels() {
  const kernelsFFT = new Array(nBins);

  for (let k = 0; k < nBins; k++) {
    const N_k = cqtKernelLengths[k];
    const f_k = cqtFrequencies[k];

    // Create Hann window
    const window = new Float32Array(N_k);
    for (let i = 0; i < N_k; i++) {
      window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (N_k - 1)));
    }

    // Create windowed complex exponential (centered)
    const kernel = new Array(cqtFFTLength);
    for (let n = 0; n < cqtFFTLength; n++) {
      kernel[n] = new Complex(0, 0);
    }

    // Compute kernel: window[n] * exp(-2πi * f_k * (n - N_k/2) / sr) / N_k
    const halfN = N_k / 2;
    for (let n = 0; n < N_k; n++) {
      const phase = -2 * Math.PI * f_k * (n - halfN) / currentSampleRate;
      const amplitude = window[n] / N_k;
      kernel[n] = new Complex(amplitude * Math.cos(phase), amplitude * Math.sin(phase));
    }

    // Compute FFT of kernel, then take conjugate for correlation
    const kernelFFT = fftComplex(kernel);
    kernelsFFT[k] = kernelFFT.map(c => c.conj());
  }

  return kernelsFFT;
}

function computeFrameCQT(frame) {
  // Zero-pad frame to FFT length
  const paddedFrame = new Float32Array(cqtFFTLength);
  const copyLength = Math.min(frame.length, cqtFFTLength);
  for (let i = 0; i < copyLength; i++) {
    paddedFrame[i] = frame[i];
  }

  // Compute FFT of frame
  const frameFFT = fft(paddedFrame);

  // Compute CQT by spectral inner product with each kernel
  const cqt = new Float32Array(nBins);

  for (let k = 0; k < nBins; k++) {
    const kernelFFTConj = cqtKernelsFFT[k];

    let sumReal = 0;
    let sumImag = 0;

    for (let n = 0; n < cqtFFTLength; n++) {
      sumReal += frameFFT[n].real * kernelFFTConj[n].real - frameFFT[n].imag * kernelFFTConj[n].imag;
      sumImag += frameFFT[n].real * kernelFFTConj[n].imag + frameFFT[n].imag * kernelFFTConj[n].real;
    }

    cqt[k] = Math.sqrt(sumReal * sumReal + sumImag * sumImag);
  }

  return cqt;
}

/**
 * Extract full CQT spectrogram for visualization
 */
function extractFullCQT(audioData) {
  if (!cqtInitialized) throw new Error('CQT not initialized');

  const numFrames = Math.floor((audioData.length - 1) / hopLength) + 1;
  const magnitudes = [];

  for (let i = 0; i < numFrames; i++) {
    const centerSample = i * hopLength;

    const startSample = Math.max(0, centerSample - Math.floor(cqtFFTLength / 2));
    const frame = new Float32Array(cqtFFTLength);

    for (let j = 0; j < cqtFFTLength; j++) {
      const idx = startSample + j;
      frame[j] = idx < audioData.length ? audioData[idx] : 0;
    }

    const frameCQT = computeFrameCQT(frame);
    magnitudes.push(frameCQT);
  }

  const times = new Float32Array(numFrames);
  for (let i = 0; i < numFrames; i++) {
    times[i] = (i * hopLength) / currentSampleRate;
  }

  return { magnitudes, times, numFrames, numBins: nBins, hopSize: hopLength };
}

/**
 * Extract CQT features for classification
 */
function extractCQTFeatures(audioData) {
  if (!cqtInitialized) throw new Error('CQT not initialized');

  const numFrames = Math.floor((audioData.length - 1) / hopLength) + 1;
  const cqt = new Float32Array(nBins * numFrames);

  for (let i = 0; i < numFrames; i++) {
    const centerSample = i * hopLength;

    const startSample = Math.max(0, centerSample - Math.floor(cqtFFTLength / 2));
    const frame = new Float32Array(cqtFFTLength);

    for (let j = 0; j < cqtFFTLength; j++) {
      const idx = startSample + j;
      frame[j] = idx < audioData.length ? audioData[idx] : 0;
    }

    const frameCQT = computeFrameCQT(frame);

    // Store in [bins, frames] format (row-major)
    for (let b = 0; b < nBins; b++) {
      cqt[b * numFrames + i] = frameCQT[b];
    }
  }

  // Normalize (min-max)
  let minVal = Infinity, maxVal = -Infinity;
  for (let i = 0; i < cqt.length; i++) {
    if (cqt[i] < minVal) minVal = cqt[i];
    if (cqt[i] > maxVal) maxVal = cqt[i];
  }
  const range = maxVal - minVal;
  if (range > 0) {
    for (let i = 0; i < cqt.length; i++) {
      cqt[i] = (cqt[i] - minVal) / range;
    }
  }

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

/**
 * Fit CQT to model input shape by padding or truncating the time dimension.
 * - If numFrames < targetFrames: zero-pad on the right
 * - If numFrames > targetFrames: truncate to first targetFrames
 * - Bins dimension is assumed to match (or handled via resizeCQTForModel).
 */
function fitCQTForModel(cqtData, targetBins, targetFrames) {
  const { cqt, numBins: srcBins, numFrames: srcFrames } = cqtData;

  // If bins don't match, resize bins via interpolation first
  if (srcBins !== targetBins) {
    return resizeCQTForModel(cqtData, targetBins, targetFrames);
  }

  if (srcFrames === targetFrames) return cqt;

  const result = new Float32Array(targetBins * targetFrames);
  const copyFrames = Math.min(srcFrames, targetFrames);

  for (let b = 0; b < targetBins; b++) {
    for (let t = 0; t < copyFrames; t++) {
      result[b * targetFrames + t] = cqt[b * srcFrames + t];
    }
    // Remaining frames (if srcFrames < targetFrames) stay as 0 (zero-padded)
  }

  return result;
}

// ============================================================================
// Onset Detection (Superflux on CQT)
// ============================================================================

/**
 * Compute onset envelope using Superflux algorithm on CQT magnitudes.
 * Based on librosa.onset.onset_strength(S=..., lag=2, max_size=3)
 *
 * @param {Float32Array[]} cqtMagnitudes - Array of CQT magnitude arrays (frame x bins)
 * @param {Object} cfg - Configuration object
 * @returns {Float32Array} - Onset strength envelope
 */
function computeSuperfluxOnsetEnvelope(cqtMagnitudes, cfg) {
  const numFrames = cqtMagnitudes.length;
  const lag = cfg.onset?.lag || 2;
  const maxSize = cfg.onset?.maxSize || 3;

  // Convert to dB (librosa.power_to_db approximation)
  // Find global max for reference
  let globalMax = 0;
  for (let t = 0; t < numFrames; t++) {
    const frame = cqtMagnitudes[t];
    for (let b = 0; b < frame.length; b++) {
      if (frame[b] > globalMax) globalMax = frame[b];
    }
  }

  // Convert to dB (10 * log10(max(val, epsilon)))
  const epsilon = 1e-10;
  const cqtDb = new Array(numFrames);
  for (let t = 0; t < numFrames; t++) {
    const frame = cqtMagnitudes[t];
    const dbFrame = new Float32Array(frame.length);
    for (let b = 0; b < frame.length; b++) {
      const val = Math.max(frame[b], epsilon) / Math.max(globalMax, epsilon);
      dbFrame[b] = 10 * Math.log10(val);
    }
    cqtDb[t] = dbFrame;
  }

  // Compute onset envelope via Superflux
  // odf[t] = sum_b max(S[t, b] - S[t-lag, b], 0)^2
  const envelope = new Float32Array(numFrames);

  for (let t = lag; t < numFrames; t++) {
    let sum = 0;
    for (let b = 0; b < cqtDb[t].length; b++) {
      const diff = cqtDb[t][b] - cqtDb[t - lag][b];
      if (diff > 0) {
        sum += diff * diff;
      }
    }
    envelope[t] = Math.sqrt(sum);
  }

  // Fill initial frames with 0 (can't compute flux before lag)
  for (let t = 0; t < lag; t++) {
    envelope[t] = 0;
  }

  // Apply temporal smoothing (max_size determines window)
  // Similar to librosa's default smoothing
  const smoothed = new Float32Array(numFrames);
  const halfWindow = Math.floor(maxSize / 2);

  for (let t = 0; t < numFrames; t++) {
    let maxVal = 0;
    const winStart = Math.max(0, t - halfWindow);
    const winEnd = Math.min(numFrames, t + halfWindow + 1);
    for (let i = winStart; i < winEnd; i++) {
      if (envelope[i] > maxVal) maxVal = envelope[i];
    }
    smoothed[t] = maxVal;
  }

  return smoothed;
}

/**
 * Peak picking with McFee parameters (best performing in tests)
 * Based on librosa.util.peak_pick
 */
function peakPickMcFee(envelope, cfg) {
  const preMax = cfg.onset?.preMax || 30;
  const postMax = cfg.onset?.postMax || 1;
  const preAvg = cfg.onset?.preAvg || 100;
  const postAvg = cfg.onset?.postAvg || 100;
  const wait = cfg.onset?.wait || 30;
  const delta = cfg.onset?.delta || 0.07;

  const numFrames = envelope.length;
  const peaks = [];

  // Compute local mean and max
  for (let i = 0; i < numFrames; i++) {
    // Check if it's a local maximum
    const maxStart = Math.max(0, i - preMax);
    const maxEnd = Math.min(numFrames, i + postMax + 1);

    let isLocalMax = true;
    for (let j = maxStart; j < maxEnd; j++) {
      if (j !== i && envelope[j] > envelope[i]) {
        isLocalMax = false;
        break;
      }
    }

    if (!isLocalMax) continue;

    // Check if above local average + delta
    const avgStart = Math.max(0, i - preAvg);
    const avgEnd = Math.min(numFrames, i + postAvg + 1);

    let sum = 0;
    for (let j = avgStart; j < avgEnd; j++) {
      sum += envelope[j];
    }
    const localAvg = sum / (avgEnd - avgStart);

    // Must be above local average + delta
    if (envelope[i] < localAvg + delta) continue;

    // Check wait period
    if (peaks.length > 0 && i - peaks[peaks.length - 1].index < wait) {
      // Replace if this peak is stronger
      if (envelope[i] > envelope[peaks[peaks.length - 1].index]) {
        peaks[peaks.length - 1] = { index: i, value: envelope[i] };
      }
    } else {
      peaks.push({ index: i, value: envelope[i] });
    }
  }

  return peaks;
}

/**
 * Detect onsets using Superflux on CQT (replaces spectral flux on raw audio)
 */
function detectOnsetsSuperflux(cqtMagnitudes, cfg) {
  const sr = cfg.audio?.sampleRate || currentSampleRate;
  const hop = cfg.audio?.hopSize || hopLength;
  const minIntervalMs = cfg.onset?.minInterval || 100;
  const minIntervalSamples = (minIntervalMs / 1000) * sr;

  // Compute onset envelope from CQT
  const envelope = computeSuperfluxOnsetEnvelope(cqtMagnitudes, cfg);

  // Peak picking
  const peaks = peakPickMcFee(envelope, cfg);

  // Convert to timestamps
  return peaks.map(p => ({
    time: (p.index * hop) / sr,
    strength: p.value,
    sample: p.index * hop
  }));
}

/**
 * Legacy onset detection (spectral flux on raw audio) - kept for reference
 * @deprecated Use detectOnsetsSuperflux instead
 */
function detectOnsetsLegacy(audioData, cfg) {
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

/**
 * Real-time onset detection on a small audio chunk
 * Simplified but fast for streaming use
 * Returns true if onset detected in the chunk
 */
function detectOnsetRealtime(audioData, threshold = 0.15) {
  const frameSize = 1024;
  const hop = 512;

  if (audioData.length < frameSize) return false;

  const numFrames = Math.floor((audioData.length - frameSize) / hop) + 1;
  if (numFrames < 2) return false;

  let prevEnergy = 0;
  let maxFlux = 0;

  for (let i = 0; i < numFrames; i++) {
    const start = i * hop;
    let energy = 0;

    for (let j = 0; j < frameSize && start + j < audioData.length; j++) {
      energy += audioData[start + j] * audioData[start + j];
    }

    if (i > 0) {
      const flux = Math.max(0, energy - prevEnergy);
      if (flux > maxFlux) maxFlux = flux;
    }

    prevEnergy = energy;
  }

  // Normalize and check threshold
  const normalizedFlux = Math.sqrt(maxFlux);
  return normalizedFlux > threshold;
}

/**
 * Process real-time audio stream for onset detection and classification
 */
async function processStreamChunk(samples, timestamp) {
  const sr = streamConfig?.sampleRate || currentSampleRate;
  const windowSize = streamConfig?.windowSize || 2.0;
  const windowSamples = Math.floor(windowSize * sr);
  const threshold = streamConfig?.threshold || 0.15;
  const minInterval = (streamConfig?.minInterval || 100) / 1000; // Convert ms to seconds
  const ignoreSubsequent = streamConfig?.ignoreSubsequent !== false;
  const flexibleWindow = streamConfig?.flexibleWindow || false;

  // If we're capturing a window, add samples to it
  if (isCapturingWindow) {
    // In flexible window mode, check for a new onset that should end the current window early
    if (flexibleWindow) {
      const currentTime = timestamp / 1000;
      const timeSinceLastOnset = currentTime - lastOnsetTime;

      if (timeSinceLastOnset >= minInterval && windowBufferSize > 0) {
        const hasNewOnset = detectOnsetRealtime(samples, threshold);

        if (hasNewOnset) {
          // Classify the current (short) buffer immediately
          await classifyAndSendWindow(flexibleWindow);

          // Start a new window with the current samples
          lastOnsetTime = currentTime;
          windowStartTime = timestamp;
          windowBuffer = new Float32Array(windowSamples);
          windowBufferSize = 0;

          const toAdd = Math.min(windowSamples, samples.length);
          windowBuffer.set(samples.subarray(0, toAdd), 0);
          windowBufferSize = toAdd;

          self.postMessage({ type: 'onset-detected', timestamp });
          return;
        }
      }
    }

    const remaining = windowSamples - windowBufferSize;
    const toAdd = Math.min(remaining, samples.length);

    windowBuffer.set(samples.subarray(0, toAdd), windowBufferSize);
    windowBufferSize += toAdd;

    // Check if window is complete (reached max size)
    if (windowBufferSize >= windowSamples) {
      await classifyAndSendWindow(flexibleWindow);

      // Reset if not ignoring subsequent
      if (!ignoreSubsequent && !flexibleWindow) {
        lastOnsetTime = -Infinity;
      }
    }

    return;
  }

  // Check for onset
  const currentTime = timestamp / 1000; // Convert to seconds
  const timeSinceLastOnset = currentTime - lastOnsetTime;

  // Only check for onset if enough time has passed
  if (timeSinceLastOnset >= minInterval) {
    const hasOnset = detectOnsetRealtime(samples, threshold);

    if (hasOnset) {
      lastOnsetTime = currentTime;
      isCapturingWindow = true;
      windowStartTime = timestamp;
      windowBuffer = new Float32Array(windowSamples);
      windowBufferSize = 0;

      // Start capturing with current samples
      const toAdd = Math.min(windowSamples, samples.length);
      windowBuffer.set(samples.subarray(0, toAdd), 0);
      windowBufferSize = toAdd;

      self.postMessage({
        type: 'onset-detected',
        timestamp: timestamp
      });
    }
  }
}

/**
 * Classify the current window buffer and send the result.
 * Supports both fixed (resize) and flexible (pad/truncate) modes.
 */
async function classifyAndSendWindow(flexibleWindow) {
  isCapturingWindow = false;

  const usedBuffer = windowBuffer.subarray(0, windowBufferSize);

  try {
    const cqtData = extractCQTFeatures(usedBuffer);
    const features = flexibleWindow
      ? fitCQTForModel(cqtData, modelInputBins, modelInputFrames)
      : resizeCQTForModel(cqtData, modelInputBins, modelInputFrames);
    const result = await predict(features);

    self.postMessage({
      type: 'stream-result',
      prediction: result,
      timestamp: windowStartTime,
      cqt: {
        data: Array.from(features),
        bins: modelInputBins,
        frames: modelInputFrames
      }
    });
  } catch (e) {
    console.error('[Worker] Classification error:', e);
  }
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

  // Extract CQT features for classification
  const cqtData = extractCQTFeatures(audioData);
  const features = resizeCQTForModel(cqtData, modelInputBins, modelInputFrames);
  const prediction = await predict(features);

  // Also extract full CQT for visualization
  const fullCQT = extractFullCQT(audioData);

  return { prediction, cqtData: fullCQT };
}

// ============================================================================
// Full Processing Pipeline
// ============================================================================

async function processAudio(audioData, cfg, audioDuration) {
  const results = { fullCQT: null, onsets: [], predictions: [] };
  const flexibleWindow = cfg.classification?.flexibleWindow || false;

  // Step 1: Extract full CQT for visualization
  reportProgress('cqt', 10, 'Extracting CQT spectrogram...');
  results.fullCQT = extractFullCQT(audioData);
  reportProgress('cqt', 30, `CQT extracted: ${results.fullCQT.numFrames} frames`);

  // Step 2: Detect onsets using Superflux on CQT
  reportProgress('onset', 35, 'Detecting onsets (Superflux)...');
  let onsets = detectOnsetsSuperflux(results.fullCQT.magnitudes, cfg);
  reportProgress('onset', 45, `Found ${onsets.length} onsets`);

  // Step 2.5: Filter subsequent onsets if enabled (skip when flexible window is on)
  if (!flexibleWindow && cfg.onset?.ignoreSubsequentOnsets) {
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

    let endSample, endTime;

    if (flexibleWindow) {
      // Flexible window: extend to next onset or end of audio
      endTime = (i < onsets.length - 1) ? onsets[i + 1].time : audioDuration;
      endSample = Math.min(Math.floor(endTime * sr), audioData.length);
    } else {
      // Fixed window: use configured window size
      endSample = Math.min(startSample + windowSamples, audioData.length);
      endTime = (i < onsets.length - 1) ? onsets[i + 1].time : audioDuration;
    }

    const windowData = audioData.slice(startSample, endSample);

    // Skip very short segments (less than 10ms worth of samples)
    const minSamples = flexibleWindow ? Math.floor(sr * 0.01) : Math.floor(windowSamples * 0.5);
    if (windowData.length < minSamples) continue;

    try {
      const cqtData = extractCQTFeatures(windowData);
      // Flexible: pad/truncate to model input; Fixed: resize via interpolation
      const features = flexibleWindow
        ? fitCQTForModel(cqtData, modelInputBins, modelInputFrames)
        : resizeCQTForModel(cqtData, modelInputBins, modelInputFrames);
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
          cfg?.classification?.cqtBins || 216,
          12,
          cfg?.audio?.hopSize || 512,
          cfg?.audio?.minFrequency || 32.70
        );

        reportProgress('init', 100, 'Ready');
        self.postMessage({ type: 'ready', id });
        break;
      }

      case 'process-audio': {
        const { audioData, config: cfg, audioDuration } = payload;
        const audioArray = audioData instanceof Float32Array ? audioData : new Float32Array(audioData);

        // Reinitialize CQT if config changed
        const newBins = cfg?.classification?.cqtBins || 216;
        const newHop = cfg?.audio?.hopSize || 512;
        const newFmin = cfg?.audio?.minFrequency || 32.70;
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
        const { prediction, cqtData } = await classifySingle(audioArray);
        self.postMessage({ type: 'result', id, prediction, cqtData });
        break;
      }

      case 'start-stream': {
        // Initialize streaming with config
        const { sampleRate, windowSize, threshold, minInterval, ignoreSubsequent, flexibleWindow } = payload || {};

        streamConfig = {
          sampleRate: sampleRate || currentSampleRate,
          windowSize: windowSize || 2.0,
          threshold: threshold || 0.15,
          minInterval: minInterval || 100,
          ignoreSubsequent: ignoreSubsequent !== false,
          flexibleWindow: flexibleWindow || false
        };

        // Reset streaming state
        isStreaming = true;
        lastOnsetTime = -Infinity;
        isCapturingWindow = false;
        windowBuffer = null;
        windowBufferSize = 0;

        console.log('[Worker] Streaming started with config:', streamConfig);
        self.postMessage({ type: 'stream-ready' });
        break;
      }

      case 'stream-detect': {
        // Real-time onset detection + classification
        if (!isStreaming) break;

        const { audioData, timestamp } = payload;
        const audioArray = audioData instanceof Float32Array ? audioData : new Float32Array(audioData);

        try {
          await processStreamChunk(audioArray, timestamp || Date.now());
        } catch (e) {
          console.error('[Worker] Stream detect error:', e);
        }
        break;
      }

      case 'stream-classify': {
        // Direct classification without onset detection (backward compatible)
        if (!isStreaming) break;
        const { audioData } = payload;
        const audioArray = audioData instanceof Float32Array ? audioData : new Float32Array(audioData);
        try {
          const { prediction, cqtData } = await classifySingle(audioArray);
          self.postMessage({ type: 'stream-result', prediction, cqtData, timestamp: Date.now() });
        } catch (e) {
          console.error('[Worker] Stream error:', e);
        }
        break;
      }

      case 'stop-stream': {
        isStreaming = false;
        isCapturingWindow = false;
        streamConfig = null;
        windowBuffer = null;
        windowBufferSize = 0;
        lastOnsetTime = -Infinity;
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
