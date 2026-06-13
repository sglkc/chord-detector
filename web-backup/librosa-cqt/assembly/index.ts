/**
 * CQT Feature Extraction Module (librosa-compatible)
 * AssemblyScript implementation for WebAssembly
 * 
 * Implements Constant-Q Transform matching librosa's cqt() output.
 */

// ============================================================================
// Memory layout for complex arrays
// We use interleaved format: [real0, imag0, real1, imag1, ...]
// ============================================================================

// Pre-allocated buffers (will be resized on init)
let kernelsFFTReal: StaticArray<f64> | null = null;  // [nBins * fftLength]
let kernelsFFTImag: StaticArray<f64> | null = null;  // [nBins * fftLength]
let kernelLengths: StaticArray<i32> | null = null;   // [nBins]
let frequencies: StaticArray<f64> | null = null;      // [nBins]
let hannWindow: StaticArray<f64> | null = null;       // [maxKernelLength]

// FFT work buffers
let fftWorkReal: StaticArray<f64> | null = null;
let fftWorkImag: StaticArray<f64> | null = null;
let bitRevTable: StaticArray<i32> | null = null;

// Pre-allocated buffers for extractFeatures (to avoid repeated allocations)
let frameCQTBuffer: StaticArray<f64> | null = null;          // [nBins]
let tempResultsBuffer: StaticArray<f64> | null = null;       // [nBins * maxTargetFrames]
let tempResultsMaxSize: i32 = 0;

// Configuration
let sampleRate: f64 = 48000.0;
let fmin: f64 = 130.81;  // C3
let nBins: i32 = 36;
let binsPerOctave: i32 = 12;
let hopLength: i32 = 512;
let Q: f64 = 0.0;
let fftLength: i32 = 0;
let maxKernelLength: i32 = 0;
let initialized: bool = false;

// ============================================================================
// Utility Functions
// ============================================================================

function nextPowerOf2(n: i32): i32 {
  let power: i32 = 1;
  while (power < n) {
    power <<= 1;
  }
  return power;
}

function log2i(n: i32): i32 {
  let result: i32 = 0;
  let val = n;
  while (val > 1) {
    val >>= 1;
    result++;
  }
  return result;
}

function bitReverse(x: i32, bits: i32): i32 {
  let result: i32 = 0;
  let val = x;
  for (let i: i32 = 0; i < bits; i++) {
    result = (result << 1) | (val & 1);
    val >>= 1;
  }
  return result;
}

// ============================================================================
// FFT Implementation (Cooley-Tukey radix-2, iterative)
// ============================================================================

/**
 * Compute FFT in-place
 * @param real - Real part array
 * @param imag - Imaginary part array
 * @param n - FFT size (must be power of 2)
 */
function fftInPlace(real: StaticArray<f64>, imag: StaticArray<f64>, n: i32): void {
  const bits = log2i(n);

  // Bit-reversal permutation
  for (let i: i32 = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    if (i < j) {
      // Swap real
      const tempReal = unchecked(real[i]);
      unchecked(real[i] = real[j]);
      unchecked(real[j] = tempReal);
      // Swap imag
      const tempImag = unchecked(imag[i]);
      unchecked(imag[i] = imag[j]);
      unchecked(imag[j] = tempImag);
    }
  }

  // Cooley-Tukey iterative FFT
  for (let size: i32 = 2; size <= n; size <<= 1) {
    const halfSize = size >> 1;
    const tableStep = n / size;
    const angleStep = -2.0 * Math.PI / <f64>size;

    for (let i: i32 = 0; i < n; i += size) {
      for (let j: i32 = 0; j < halfSize; j++) {
        const theta = angleStep * <f64>j;
        const twiddleReal = Math.cos(theta);
        const twiddleImag = Math.sin(theta);

        const idx1 = i + j;
        const idx2 = i + j + halfSize;

        const aReal = unchecked(real[idx1]);
        const aImag = unchecked(imag[idx1]);
        const bReal = unchecked(real[idx2]);
        const bImag = unchecked(imag[idx2]);

        // b * twiddle
        const tReal = bReal * twiddleReal - bImag * twiddleImag;
        const tImag = bReal * twiddleImag + bImag * twiddleReal;

        // Butterfly
        unchecked(real[idx1] = aReal + tReal);
        unchecked(imag[idx1] = aImag + tImag);
        unchecked(real[idx2] = aReal - tReal);
        unchecked(imag[idx2] = aImag - tImag);
      }
    }
  }
}

// ============================================================================
// Window Functions
// ============================================================================

function computeHannWindow(length: i32): StaticArray<f64> {
  const window = new StaticArray<f64>(length);
  const lengthMinus1 = <f64>(length - 1);
  for (let i: i32 = 0; i < length; i++) {
    unchecked(window[i] = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * <f64>i / lengthMinus1)));
  }
  return window;
}

// ============================================================================
// CQT Kernel Computation
// ============================================================================

function computeKernels(): void {
  // Allocate kernel storage
  const totalKernelSize = nBins * fftLength;
  kernelsFFTReal = new StaticArray<f64>(totalKernelSize);
  kernelsFFTImag = new StaticArray<f64>(totalKernelSize);

  // Temporary buffers for kernel FFT computation
  const kernelReal = new StaticArray<f64>(fftLength);
  const kernelImag = new StaticArray<f64>(fftLength);

  for (let k: i32 = 0; k < nBins; k++) {
    const N_k = unchecked(kernelLengths![k]);
    const f_k = unchecked(frequencies![k]);

    // Initialize with zeros
    for (let n: i32 = 0; n < fftLength; n++) {
      unchecked(kernelReal[n] = 0.0);
      unchecked(kernelImag[n] = 0.0);
    }

    // Compute kernel: window[n] * exp(-2πi * f_k * (n - N_k/2) / sr) / N_k
    const halfN = <f64>N_k / 2.0;
    const invN_k = 1.0 / <f64>N_k;
    const phaseMultiplier = -2.0 * Math.PI * f_k / sampleRate;

    // Compute Hann window for this kernel length
    const windowLengthMinus1 = <f64>(N_k - 1);

    for (let n: i32 = 0; n < N_k; n++) {
      const windowVal = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * <f64>n / windowLengthMinus1));
      const phase = phaseMultiplier * (<f64>n - halfN);
      const amplitude = windowVal * invN_k;
      unchecked(kernelReal[n] = amplitude * Math.cos(phase));
      unchecked(kernelImag[n] = amplitude * Math.sin(phase));
    }

    // Compute FFT of kernel
    fftInPlace(kernelReal, kernelImag, fftLength);

    // Store conjugate of kernel FFT
    const baseIdx = k * fftLength;
    for (let n: i32 = 0; n < fftLength; n++) {
      unchecked(kernelsFFTReal![baseIdx + n] = kernelReal[n]);
      unchecked(kernelsFFTImag![baseIdx + n] = -kernelImag[n]);  // Conjugate
    }
  }
}

// ============================================================================
// Exported Functions
// ============================================================================

/**
 * Initialize the CQT extractor with parameters
 */
export function init(
  _sampleRate: f64,
  _fmin: f64,
  _nBins: i32,
  _binsPerOctave: i32,
  _hopLength: i32
): void {
  sampleRate = _sampleRate;
  fmin = _fmin;
  nBins = _nBins;
  binsPerOctave = _binsPerOctave;
  hopLength = _hopLength;

  // Calculate Q factor
  // Q = 1 / (2^(1/bins_per_octave) - 1)
  Q = 1.0 / (Math.pow(2.0, 1.0 / <f64>binsPerOctave) - 1.0);

  // Calculate center frequencies
  frequencies = new StaticArray<f64>(nBins);
  for (let k: i32 = 0; k < nBins; k++) {
    unchecked(frequencies![k] = fmin * Math.pow(2.0, <f64>k / <f64>binsPerOctave));
  }

  // Calculate window lengths for each bin
  kernelLengths = new StaticArray<i32>(nBins);
  maxKernelLength = 0;
  for (let k: i32 = 0; k < nBins; k++) {
    const length = <i32>Math.ceil(Q * sampleRate / unchecked(frequencies![k]));
    unchecked(kernelLengths![k] = length);
    if (length > maxKernelLength) {
      maxKernelLength = length;
    }
  }

  // Determine FFT size
  fftLength = nextPowerOf2(maxKernelLength);

  // Allocate FFT work buffers
  fftWorkReal = new StaticArray<f64>(fftLength);
  fftWorkImag = new StaticArray<f64>(fftLength);

  // Allocate frame CQT buffer (for extractFeatures)
  frameCQTBuffer = new StaticArray<f64>(nBins);

  // Pre-compute CQT kernels
  computeKernels();

  initialized = true;
}

/**
 * Get the FFT length (needed for buffer allocation on JS side)
 */
export function getFFTLength(): i32 {
  return fftLength;
}

/**
 * Get number of bins
 */
export function getNumBins(): i32 {
  return nBins;
}

/**
 * Get center frequency for a specific bin
 */
export function getFrequency(bin: i32): f64 {
  if (frequencies === null || bin < 0 || bin >= nBins) return 0.0;
  return unchecked(frequencies![bin]);
}

/**
 * Compute CQT for a single frame
 * @param framePtr - Pointer to frame data (Float64Array in WASM memory)
 * @param frameLength - Length of frame data
 * @param outputPtr - Pointer to output buffer (Float64Array, length = nBins)
 */
export function computeFrameCQT(
  framePtr: usize,
  frameLength: i32,
  outputPtr: usize
): void {
  if (!initialized || fftWorkReal === null || fftWorkImag === null) return;

  // Zero-fill work buffers and copy frame data
  for (let i: i32 = 0; i < fftLength; i++) {
    if (i < frameLength) {
      unchecked(fftWorkReal![i] = load<f64>(framePtr + (<usize>i << 3)));
    } else {
      unchecked(fftWorkReal![i] = 0.0);
    }
    unchecked(fftWorkImag![i] = 0.0);
  }

  // Compute FFT of frame
  fftInPlace(fftWorkReal!, fftWorkImag!, fftLength);

  // Compute CQT by spectral inner product with each kernel
  for (let k: i32 = 0; k < nBins; k++) {
    const kernelBaseIdx = k * fftLength;
    let sumReal: f64 = 0.0;
    let sumImag: f64 = 0.0;

    for (let n: i32 = 0; n < fftLength; n++) {
      const frameReal = unchecked(fftWorkReal![n]);
      const frameImag = unchecked(fftWorkImag![n]);
      const kernelReal = unchecked(kernelsFFTReal![kernelBaseIdx + n]);
      const kernelImag = unchecked(kernelsFFTImag![kernelBaseIdx + n]);

      // Complex multiplication: frame * kernelConj
      sumReal += frameReal * kernelReal - frameImag * kernelImag;
      sumImag += frameReal * kernelImag + frameImag * kernelReal;
    }

    // Store magnitude
    const magnitude = Math.sqrt(sumReal * sumReal + sumImag * sumImag);
    store<f64>(outputPtr + (<usize>k << 3), magnitude);
  }
}

/**
 * Extract CQT features for classification
 * Processes audio and outputs normalized features in [bins, frames] format
 * 
 * @param audioPtr - Pointer to audio data (Float64Array)
 * @param audioLength - Length of audio data
 * @param targetFrames - Number of target frames for output
 * @param outputPtr - Pointer to output buffer (Float64Array, length = nBins * targetFrames)
 */
export function extractFeatures(
  audioPtr: usize,
  audioLength: i32,
  targetFrames: i32,
  outputPtr: usize
): void {
  if (!initialized || frameCQTBuffer === null) return;

  // Calculate number of frames
  const numFrames = (audioLength - 1) / hopLength + 1;
  const actualFrames = numFrames < targetFrames ? numFrames : targetFrames;

  const totalSize = nBins * targetFrames;

  // Ensure tempResultsBuffer is allocated and large enough
  if (tempResultsBuffer === null || tempResultsMaxSize < totalSize) {
    tempResultsBuffer = new StaticArray<f64>(totalSize);
    tempResultsMaxSize = totalSize;
  }

  // Track min/max for normalization
  let minVal: f64 = Infinity;
  let maxVal: f64 = -Infinity;

  // First pass: compute CQT and find min/max
  for (let i: i32 = 0; i < targetFrames; i++) {
    if (i < actualFrames) {
      const centerSample = i * hopLength;
      const halfFFT = fftLength >> 1;
      const startSample = centerSample > halfFFT ? centerSample - halfFFT : 0;

      // Extract and process frame
      for (let j: i32 = 0; j < fftLength; j++) {
        const idx = startSample + j;
        if (idx < audioLength) {
          unchecked(fftWorkReal![j] = load<f64>(audioPtr + (<usize>idx << 3)));
        } else {
          unchecked(fftWorkReal![j] = 0.0);
        }
        unchecked(fftWorkImag![j] = 0.0);
      }

      // FFT
      fftInPlace(fftWorkReal!, fftWorkImag!, fftLength);

      // CQT computation
      for (let k: i32 = 0; k < nBins; k++) {
        const kernelBaseIdx = k * fftLength;
        let sumReal: f64 = 0.0;
        let sumImag: f64 = 0.0;

        for (let n: i32 = 0; n < fftLength; n++) {
          const frameReal = unchecked(fftWorkReal![n]);
          const frameImag = unchecked(fftWorkImag![n]);
          const kernelReal = unchecked(kernelsFFTReal![kernelBaseIdx + n]);
          const kernelImag = unchecked(kernelsFFTImag![kernelBaseIdx + n]);

          sumReal += frameReal * kernelReal - frameImag * kernelImag;
          sumImag += frameReal * kernelImag + frameImag * kernelReal;
        }

        const magnitude = Math.sqrt(sumReal * sumReal + sumImag * sumImag);
        unchecked(frameCQTBuffer![k] = magnitude);

        if (magnitude < minVal) minVal = magnitude;
        if (magnitude > maxVal) maxVal = magnitude;
      }

      // Store in [bins, frames] order
      for (let k: i32 = 0; k < nBins; k++) {
        unchecked(tempResultsBuffer![k * targetFrames + i] = frameCQTBuffer![k]);
      }
    } else {
      // Pad with zeros
      for (let k: i32 = 0; k < nBins; k++) {
        unchecked(tempResultsBuffer![k * targetFrames + i] = 0.0);
      }
    }
  }

  // Second pass: normalize and store to output
  const range = maxVal - minVal;
  const invRange = range > 0.0 ? 1.0 / range : 0.0;

  for (let i: i32 = 0; i < totalSize; i++) {
    const val = unchecked(tempResultsBuffer![i]);
    const normalized = range > 0.0 ? (val - minVal) * invRange : 0.0;
    store<f64>(outputPtr + (<usize>i << 3), normalized);
  }
}

/**
 * Memory allocation helper - allocate Float64Array
 */
export function allocateF64Array(length: i32): usize {
  const arr = new StaticArray<f64>(length);
  return changetype<usize>(arr);
}

/**
 * Memory deallocation helper
 */
export function deallocate(ptr: usize): void {
  // AssemblyScript's GC will handle this
  // This is a no-op but provides API consistency
}
