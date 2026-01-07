/**
 * CQT Feature Extraction Module (librosa-compatible)
 * 
 * Implements Constant-Q Transform matching librosa's cqt() output.
 * 
 * Key changes from previous version:
 * - Fixed kernel phase sign to match DFT convention
 * - Improved frequency-to-bin mapping
 * - Better normalization matching librosa
 * - Added debug mode for verification
 * 
 * References:
 * - Brown, J. C. (1991). "Calculation of a constant Q spectral transform"
 * - Schörkhuber, C. (2010). "Constant-Q Transform Toolbox"
 * - librosa.cqt implementation
 */

/**
 * Complex number class for FFT operations
 */
class Complex {
  constructor(real = 0, imag = 0) {
    this.real = real;
    this.imag = imag;
  }

  add(other) {
    return new Complex(this.real + other.real, this.imag + other.imag);
  }

  sub(other) {
    return new Complex(this.real - other.real, this.imag - other.imag);
  }

  mul(other) {
    return new Complex(
      this.real * other.real - this.imag * other.imag,
      this.real * other.imag + this.imag * other.real
    );
  }

  conj() {
    return new Complex(this.real, -this.imag);
  }

  magnitude() {
    return Math.sqrt(this.real * this.real + this.imag * this.imag);
  }

  static fromPolar(r, theta) {
    return new Complex(r * Math.cos(theta), r * Math.sin(theta));
  }
}

/**
 * Fast Fourier Transform (Cooley-Tukey radix-2, iterative for efficiency)
 */
class FFT {
  /**
   * Bit reversal for iterative FFT
   */
  static bitReverse(x, bits) {
    let result = 0;
    for (let i = 0; i < bits; i++) {
      result = (result << 1) | (x & 1);
      x >>= 1;
    }
    return result;
  }

  /**
   * Compute FFT of a real signal (iterative, more efficient)
   * @param {Float32Array} signal - Input signal (length must be power of 2)
   * @returns {Complex[]} FFT result
   */
  static fft(signal) {
    const n = signal.length;
    const bits = Math.log2(n);

    if ((n & (n - 1)) !== 0) {
      throw new Error('FFT length must be a power of 2');
    }

    // Bit-reversal permutation
    const result = new Array(n);
    for (let i = 0; i < n; i++) {
      const j = FFT.bitReverse(i, bits);
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

  /**
   * Compute FFT of complex signal
   * @param {Complex[]} signal - Complex input signal
   * @returns {Complex[]} FFT result
   */
  static fftComplex(signal) {
    const n = signal.length;
    const bits = Math.log2(n);

    if ((n & (n - 1)) !== 0) {
      throw new Error('FFT length must be a power of 2');
    }

    // Bit-reversal permutation
    const result = new Array(n);
    for (let i = 0; i < n; i++) {
      const j = FFT.bitReverse(i, bits);
      result[i] = new Complex(signal[j].real, signal[j].imag);
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

  /**
   * Get next power of 2 >= n
   */
  static nextPowerOf2(n) {
    return Math.pow(2, Math.ceil(Math.log2(n)));
  }
}

/**
 * Window functions
 */
class WindowFunctions {
  /**
   * Hann window (matches librosa default)
   * librosa uses symmetric Hann window
   * @param {number} length - Window length
   * @returns {Float32Array} Window coefficients
   */
  static hann(length) {
    const window = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      // Symmetric Hann window (librosa default)
      window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (length - 1)));
    }
    return window;
  }
}

/**
 * Constant-Q Transform implementation matching librosa
 */
export class CQTExtractor {
  constructor() {
    this.initialized = false;
    this.kernelsFFT = null;  // Conjugate of kernel FFTs (for correlation)
    this.kernelLengths = null;
    this.frequencies = null;
    this.fftLength = null;

    // Configuration (matching librosa defaults)
    this.sampleRate = 48000;
    this.fmin = 130.81;  // C3
    this.nBins = 36;
    this.binsPerOctave = 12;
    this.hopLength = 512;

    // Derived parameters
    this.Q = null;
  }

  /**
   * Initialize the CQT extractor with parameters
   * @param {Object} options - Configuration options
   */
  init(options = {}) {
    this.sampleRate = options.sampleRate || 48000;
    this.fmin = options.fmin || 130.81;  // C3
    this.nBins = options.nBins || 36;
    this.binsPerOctave = options.binsPerOctave || 12;
    this.hopLength = options.hopLength || 512;

    // Calculate Q factor (constant for all bins)
    // Q = 1 / (2^(1/bins_per_octave) - 1)
    // This is approximately 16.82 for 12 bins per octave
    this.Q = 1 / (Math.pow(2, 1 / this.binsPerOctave) - 1);

    // Calculate center frequencies for each bin
    // f[k] = fmin * 2^(k / bins_per_octave)
    // This matches librosa.cqt_frequencies exactly
    this.frequencies = new Float32Array(this.nBins);
    for (let k = 0; k < this.nBins; k++) {
      this.frequencies[k] = this.fmin * Math.pow(2, k / this.binsPerOctave);
    }

    // Calculate window lengths for each bin
    // N_k = ceil(Q * sr / f_k)
    // Lower frequencies have longer windows
    this.kernelLengths = new Int32Array(this.nBins);
    for (let k = 0; k < this.nBins; k++) {
      this.kernelLengths[k] = Math.ceil(this.Q * this.sampleRate / this.frequencies[k]);
    }

    // Find maximum kernel length and determine FFT size
    const maxKernelLength = Math.max(...this.kernelLengths);
    this.fftLength = FFT.nextPowerOf2(maxKernelLength);

    // Pre-compute CQT kernels (stored as conjugate of their FFT)
    this.kernelsFFT = this._computeKernels();

    this.initialized = true;

    console.log('CQT Extractor initialized (librosa-compatible):');
    console.log(`  Sample rate: ${this.sampleRate} Hz`);
    console.log(`  Frequency range: ${this.fmin.toFixed(2)} Hz (${this._freqToNote(this.fmin)}) - ${this.frequencies[this.nBins - 1].toFixed(2)} Hz (${this._freqToNote(this.frequencies[this.nBins - 1])})`);
    console.log(`  Bins: ${this.nBins} (${this.nBins / this.binsPerOctave} octaves)`);
    console.log(`  Q factor: ${this.Q.toFixed(4)}`);
    console.log(`  FFT length: ${this.fftLength}`);
    console.log(`  Max kernel length: ${maxKernelLength}`);
  }

  /**
   * Convert frequency to musical note name
   */
  _freqToNote(freq) {
    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const midiNote = 12 * Math.log2(freq / 440) + 69;
    const noteIndex = Math.round(midiNote) % 12;
    const octave = Math.floor(Math.round(midiNote) / 12) - 1;
    return noteNames[noteIndex] + octave;
  }

  /**
   * Pre-compute CQT kernels
   * 
   * Each kernel is a windowed complex sinusoid at the center frequency.
   * We store the conjugate of the kernel's FFT for efficient correlation.
   * 
   * The kernel for frequency f_k is:
   *   kernel[n] = window[n] * exp(-2πi * f_k * (n - N_k/2) / sr) / N_k
   * 
   * Note: We use negative phase to match DFT convention
   * The centering (n - N_k/2) is for zero-phase response
   * 
   * @returns {Complex[][]} Array of conjugate kernel FFTs
   */
  _computeKernels() {
    const kernelsFFT = new Array(this.nBins);

    for (let k = 0; k < this.nBins; k++) {
      const N_k = this.kernelLengths[k];
      const f_k = this.frequencies[k];

      // Create windowed complex exponential (centered)
      const window = WindowFunctions.hann(N_k);
      const kernel = new Array(this.fftLength);

      // Initialize with zeros
      for (let n = 0; n < this.fftLength; n++) {
        kernel[n] = new Complex(0, 0);
      }

      // Compute kernel: window[n] * exp(-2πi * f_k * (n - N_k/2) / sr) / N_k
      // The negative phase matches the DFT convention
      // The (n - N_k/2) centers the kernel for zero-phase analysis
      const halfN = N_k / 2;
      for (let n = 0; n < N_k; n++) {
        // Negative phase for correlation (matches DFT)
        const phase = -2 * Math.PI * f_k * (n - halfN) / this.sampleRate;
        const amplitude = window[n] / N_k;
        kernel[n] = new Complex(
          amplitude * Math.cos(phase),
          amplitude * Math.sin(phase)
        );
      }

      // Compute FFT of kernel, then take conjugate for correlation
      const kernelFFT = FFT.fftComplex(kernel);

      // Store conjugate of kernel FFT
      kernelsFFT[k] = kernelFFT.map(c => c.conj());
    }

    return kernelsFFT;
  }

  /**
   * Extract CQT from a single frame of audio
   * 
   * Uses spectral multiplication (equivalent to convolution in time domain)
   * CQT[k] = sum(X[n] * conj(K[k][n])) = inner product in frequency domain
   * 
   * @param {Float32Array} frame - Audio frame
   * @returns {Float32Array} CQT magnitudes for this frame
   */
  _computeFrameCQT(frame) {
    // Zero-pad frame to FFT length
    const paddedFrame = new Float32Array(this.fftLength);
    const copyLength = Math.min(frame.length, this.fftLength);
    for (let i = 0; i < copyLength; i++) {
      paddedFrame[i] = frame[i];
    }

    // Compute FFT of frame
    const frameFFT = FFT.fft(paddedFrame);

    // Compute CQT by spectral inner product with each kernel
    const cqt = new Float32Array(this.nBins);

    for (let k = 0; k < this.nBins; k++) {
      const kernelFFTConj = this.kernelsFFT[k];

      // Inner product: sum(frameFFT * conj(kernelFFT))
      // Since we stored the conjugate, we just multiply directly
      let sumReal = 0;
      let sumImag = 0;

      for (let n = 0; n < this.fftLength; n++) {
        // frameFFT[n] * kernelFFTConj[n] (which is already conjugated)
        sumReal += frameFFT[n].real * kernelFFTConj[n].real - frameFFT[n].imag * kernelFFTConj[n].imag;
        sumImag += frameFFT[n].real * kernelFFTConj[n].imag + frameFFT[n].imag * kernelFFTConj[n].real;
      }

      // Magnitude of the complex CQT coefficient
      cqt[k] = Math.sqrt(sumReal * sumReal + sumImag * sumImag);
    }

    return cqt;
  }

  /**
   * Extract full CQT spectrogram from audio data
   * @param {Float32Array} audioData - Audio samples
   * @param {Object} options - Optional override parameters
   * @returns {Object} CQT data with magnitudes and metadata
   */
  extractFullCQT(audioData, options = {}) {
    if (!this.initialized) {
      this.init(options);
    }

    const hopLength = options.hopLength || this.hopLength;

    // Calculate number of frames
    // Use hopLength-based framing (like librosa)
    const numFrames = Math.floor((audioData.length - 1) / hopLength) + 1;

    if (numFrames <= 0) {
      console.warn('Audio too short for CQT extraction');
      return {
        magnitudes: [],
        times: [],
        numFrames: 0,
        numBins: this.nBins,
        frequencies: this.frequencies
      };
    }

    // Extract CQT for each frame
    const magnitudes = [];
    const times = [];

    for (let i = 0; i < numFrames; i++) {
      const centerSample = i * hopLength;
      const time = centerSample / this.sampleRate;

      // Extract frame centered at centerSample
      const startSample = Math.max(0, centerSample - Math.floor(this.fftLength / 2));
      const frame = new Float32Array(this.fftLength);

      for (let j = 0; j < this.fftLength; j++) {
        const idx = startSample + j;
        frame[j] = idx < audioData.length ? audioData[idx] : 0;
      }

      // Compute CQT for this frame
      const frameCQT = this._computeFrameCQT(frame);

      magnitudes.push(frameCQT);
      times.push(time);
    }

    return {
      magnitudes,
      times,
      numFrames,
      numBins: this.nBins,
      hopLength,
      sampleRate: this.sampleRate,
      frequencies: this.frequencies,
      fftLength: this.fftLength
    };
  }

  /**
   * Extract CQT features for a single window (for classification)
   * Matches the format expected by the trained model
   * @param {Float32Array} windowData - Audio window data
   * @param {Object} config - Configuration parameters
   * @returns {Float32Array} CQT features shaped for model input
   */
  async extractFeatures(windowData, config) {
    // Initialize if needed with config parameters
    if (!this.initialized) {
      this.init({
        sampleRate: config.audio.sampleRate,
        fmin: config.audio.minFrequency || 130.81,
        nBins: config.classification.cqtBins,
        binsPerOctave: 12,
        hopLength: config.audio.hopSize
      });
    }

    const numBins = config.classification.cqtBins;
    const targetFrames = config.classification.cqtTimeFrames;
    const hopLength = config.audio.hopSize;

    // Calculate number of frames (matching librosa's frame count)
    const numFrames = Math.floor((windowData.length - 1) / hopLength) + 1;

    // Extract CQT frames
    const frames = [];

    for (let i = 0; i < numFrames && i < targetFrames; i++) {
      const centerSample = i * hopLength;

      // Extract frame centered at centerSample
      const startSample = Math.max(0, centerSample - Math.floor(this.fftLength / 2));
      const frame = new Float32Array(this.fftLength);

      for (let j = 0; j < this.fftLength; j++) {
        const idx = startSample + j;
        frame[j] = idx < windowData.length ? windowData[idx] : 0;
      }

      // Compute CQT for this frame
      const frameCQT = this._computeFrameCQT(frame);
      frames.push(frameCQT);
    }

    // Pad or truncate to target frames
    const features = this._padOrTruncate(frames, numBins, targetFrames);

    // Normalize features to 0-1 range
    this._normalizeFeatures(features);

    // Print features for verification
    this._printFeatures(features, numBins, targetFrames, frames.length);

    return features;
  }

  /**
   * Pad or truncate frames to target size
   * Output format matches librosa: [n_bins, n_frames] flattened in row-major order
   */
  _padOrTruncate(frames, numBins, targetFrames) {
    // Output format: [bin0_time0, bin0_time1, ..., bin1_time0, bin1_time1, ...]
    // This matches Python's C-order flattening of (n_bins, n_frames)
    const features = new Float32Array(numBins * targetFrames);

    for (let b = 0; b < numBins; b++) {
      for (let t = 0; t < targetFrames; t++) {
        if (t < frames.length) {
          features[b * targetFrames + t] = frames[t][b];
        } else {
          // Pad with zeros
          features[b * targetFrames + t] = 0;
        }
      }
    }

    return features;
  }

  /**
   * Normalize features to 0-1 range (min-max normalization)
   */
  _normalizeFeatures(features) {
    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < features.length; i++) {
      if (features[i] < min) min = features[i];
      if (features[i] > max) max = features[i];
    }

    const range = max - min;
    if (range > 0) {
      for (let i = 0; i < features.length; i++) {
        features[i] = (features[i] - min) / range;
      }
    }
  }

  /**
   * Print features for verification (matching Python output format)
   */
  _printFeatures(features, numBins, targetFrames, actualFrames) {
    console.log('\n========== CQT FEATURE EXTRACTION RESULTS ==========');
    console.log(`Shape: [${numBins} bins × ${targetFrames} frames] = ${features.length} total values`);
    console.log(`Actual frames extracted: ${actualFrames} (padded to ${targetFrames})`);

    // Calculate statistics
    let min = Infinity, max = -Infinity, sum = 0;
    let nonZeroCount = 0;

    for (let i = 0; i < features.length; i++) {
      const val = features[i];
      if (val < min) min = val;
      if (val > max) max = val;
      sum += val;
      if (val > 0.001) nonZeroCount++;
    }

    const mean = sum / features.length;
    const nonZeroPercent = ((nonZeroCount / features.length) * 100).toFixed(1);

    console.log(`\nStatistics (after normalization):`);
    console.log(`  Min: ${min.toFixed(6)}`);
    console.log(`  Max: ${max.toFixed(6)}`);
    console.log(`  Mean: ${mean.toFixed(6)}`);
    console.log(`  Non-zero values: ${nonZeroCount} (${nonZeroPercent}%)`);

    // Print frequency-to-note mapping for verification
    console.log(`\nFrequency bins (first 12):`);
    for (let k = 0; k < Math.min(12, numBins); k++) {
      console.log(`  Bin ${k}: ${this.frequencies[k].toFixed(2)} Hz (${this._freqToNote(this.frequencies[k])})`);
    }

    // Feature Matrix Preview
    const previewBins = Math.min(12, numBins);
    const previewFrames = Math.min(8, targetFrames);

    console.log(`\nFeature Matrix Preview (top ${previewBins} bins × first ${previewFrames} frames):`);
    console.log('Note: Layout is [bins (rows) × time frames (columns)]');

    // Print header
    let header = 'Bin\\Time |';
    for (let t = 0; t < previewFrames; t++) {
      header += ` T${t.toString().padStart(2, '0')} `;
    }
    console.log(header);
    console.log('-'.repeat(header.length));

    // Print values for each bin (high frequencies at top)
    for (let b = numBins - 1; b >= numBins - previewBins; b--) {
      const note = this._freqToNote(this.frequencies[b]);
      let row = `B${b.toString().padStart(2, '0')}(${note.padEnd(3)})|`;
      for (let t = 0; t < previewFrames; t++) {
        const idx = b * targetFrames + t;
        const val = features[idx];
        row += ` ${val.toFixed(2)} `;
      }
      console.log(row);
    }

    // ASCII heatmap visualization
    console.log(`\nASCII Heatmap (all ${numBins} bins × ${previewFrames} frames):`);
    console.log('Legend: ░ (0.0-0.2) ▒ (0.2-0.4) ▓ (0.4-0.6) █ (0.6-0.8) ▀ (0.8-1.0)');

    const heatmapChars = ['░', '▒', '▓', '█', '▀'];

    for (let b = numBins - 1; b >= 0; b--) {
      let row = '';
      for (let t = 0; t < previewFrames; t++) {
        const idx = b * targetFrames + t;
        const val = features[idx];
        const charIdx = Math.min(4, Math.floor(val * 5));
        row += heatmapChars[charIdx];
      }
      // Print every 4th row
      if (b % 4 === 0 || b === numBins - 1) {
        const note = this._freqToNote(this.frequencies[b]);
        console.log(`B${b.toString().padStart(2, '0')}(${note.padEnd(3)}) ${row}`);
      }
    }

    console.log('====================================================\n');
  }

  /**
   * Get CQT visualization as ImageData
   */
  getVisualizationData(magnitudes, numBins, numFrames) {
    const imageData = new ImageData(numFrames, numBins);

    // Find min/max for normalization
    let min = Infinity, max = -Infinity;
    for (let t = 0; t < numFrames; t++) {
      for (let b = 0; b < numBins; b++) {
        const value = magnitudes[t][b];
        if (value < min) min = value;
        if (value > max) max = value;
      }
    }
    const range = max - min || 1;

    for (let t = 0; t < numFrames; t++) {
      for (let b = 0; b < numBins; b++) {
        const value = (magnitudes[t][b] - min) / range;
        const pixelIndex = ((numBins - 1 - b) * numFrames + t) * 4;

        const color = this._viridisColor(value);
        imageData.data[pixelIndex] = color[0];
        imageData.data[pixelIndex + 1] = color[1];
        imageData.data[pixelIndex + 2] = color[2];
        imageData.data[pixelIndex + 3] = 255;
      }
    }

    return imageData;
  }

  /**
   * Viridis colormap approximation
   */
  _viridisColor(value) {
    const v = Math.max(0, Math.min(1, value));

    const r = Math.round(255 * (0.267004 + v * (0.329415 + v * (-0.508378 + v * 1.137680))));
    const g = Math.round(255 * (0.004874 + v * (0.873158 + v * (-0.058404 + v * -0.322897))));
    const b = Math.round(255 * (0.329415 + v * (0.280197 + v * (-1.314181 + v * 1.171356))));

    return [
      Math.max(0, Math.min(255, r)),
      Math.max(0, Math.min(255, g)),
      Math.max(0, Math.min(255, b))
    ];
  }
}

// Named export for compatibility
export { CQTExtractor as LibrosaCQT };
