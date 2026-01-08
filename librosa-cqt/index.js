/**
 * CQT Feature Extraction Module (librosa-compatible)
 * JavaScript wrapper for AssemblyScript/WebAssembly implementation
 * 
 * Provides the same API as the pure JavaScript version but uses
 * WebAssembly for high-performance computation.
 */

// Import the auto-generated WASM bindings
import * as wasmRelease from './build/release.js';

/**
 * CQT Extractor class - JavaScript wrapper for WASM implementation
 */
export class CQTExtractor {
  constructor() {
    this.initialized = false;
    this.wasm = wasmRelease;
    this.config = null;

    // Cached pointers for memory management
    this.audioPtr = 0;
    this.outputPtr = 0;
    this.framePtr = 0;
    this.allocatedAudioLength = 0;
    this.allocatedOutputLength = 0;
    this.allocatedFrameLength = 0;
  }

  /**
   * Initialize the CQT extractor with parameters
   * @param {Object} options - Configuration options
   */
  init(options = {}) {
    // Configuration (matching librosa defaults)
    const sampleRate = options.sampleRate || 48000;
    const fmin = options.fmin || 130.81;  // C3
    const nBins = options.nBins || 36;
    const binsPerOctave = options.binsPerOctave || 12;
    const hopLength = options.hopLength || 512;

    this.config = { sampleRate, fmin, nBins, binsPerOctave, hopLength };

    // Initialize WASM CQT
    this.wasm.init(sampleRate, fmin, nBins, binsPerOctave, hopLength);

    this.initialized = true;

    const fftLength = this.wasm.getFFTLength();
    const maxFreq = fmin * Math.pow(2, (nBins - 1) / binsPerOctave);

    console.log('CQT Extractor initialized (WASM):');
    console.log(`  Sample rate: ${sampleRate} Hz`);
    console.log(`  Frequency range: ${fmin.toFixed(2)} Hz - ${maxFreq.toFixed(2)} Hz`);
    console.log(`  Bins: ${nBins} (${nBins / binsPerOctave} octaves)`);
    console.log(`  FFT length: ${fftLength}`);

    return this;
  }

  /**
   * Get WASM memory as a typed array view
   */
  _getMemoryF64() {
    return new Float64Array(this.wasm.memory.buffer);
  }

  /**
   * Allocate memory in WASM for Float64Array
   */
  _allocateF64(length) {
    return this.wasm.allocateF64Array(length);
  }

  /**
   * Copy Float32Array to WASM memory as Float64
   */
  _copyToWasm(data, ptr) {
    const memory = this._getMemoryF64();
    const offset = ptr / 8;  // Convert byte offset to f64 offset
    for (let i = 0; i < data.length; i++) {
      memory[offset + i] = data[i];
    }
  }

  /**
   * Copy from WASM memory to Float32Array
   */
  _copyFromWasm(ptr, length) {
    const memory = this._getMemoryF64();
    const offset = ptr / 8;
    const result = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      result[i] = memory[offset + i];
    }
    return result;
  }

  /**
   * Extract CQT features for a single window (for classification)
   * @param {Float32Array} windowData - Audio window data
   * @param {Object} config - Configuration parameters
   * @returns {Promise<Float32Array>} CQT features shaped for model input
   */
  async extractFeatures(windowData, config) {
    // Initialize if needed
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
    const audioLength = windowData.length;
    const outputLength = numBins * targetFrames;

    // Allocate/reuse WASM memory for audio input
    if (this.allocatedAudioLength < audioLength) {
      if (this.audioPtr) this.wasm.deallocate(this.audioPtr);
      this.audioPtr = this._allocateF64(audioLength);
      this.allocatedAudioLength = audioLength;
    }

    // Allocate/reuse WASM memory for output
    if (this.allocatedOutputLength < outputLength) {
      if (this.outputPtr) this.wasm.deallocate(this.outputPtr);
      this.outputPtr = this._allocateF64(outputLength);
      this.allocatedOutputLength = outputLength;
    }

    // Copy audio to WASM memory
    this._copyToWasm(windowData, this.audioPtr);

    // Call WASM extraction function
    this.wasm.extractFeatures(
      this.audioPtr,
      audioLength,
      targetFrames,
      this.outputPtr
    );

    // Copy results back
    const features = this._copyFromWasm(this.outputPtr, outputLength);

    // Print features for verification
    this._printFeatures(features, numBins, targetFrames);

    return features;
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

    const hopLength = options.hopLength || this.config.hopLength;
    const numBins = this.config.nBins;
    const fftLength = this.wasm.getFFTLength();

    // Calculate number of frames
    const numFrames = Math.floor((audioData.length - 1) / hopLength) + 1;

    if (numFrames <= 0) {
      console.warn('Audio too short for CQT extraction');
      return {
        magnitudes: [],
        times: [],
        numFrames: 0,
        numBins,
        frequencies: this._getFrequencies()
      };
    }

    // Allocate buffers
    const audioLength = audioData.length;
    if (this.allocatedAudioLength < fftLength) {
      if (this.audioPtr) this.wasm.deallocate(this.audioPtr);
      this.audioPtr = this._allocateF64(fftLength);
      this.allocatedAudioLength = fftLength;
    }

    if (this.allocatedFrameLength < numBins) {
      if (this.framePtr) this.wasm.deallocate(this.framePtr);
      this.framePtr = this._allocateF64(numBins);
      this.allocatedFrameLength = numBins;
    }

    const magnitudes = [];
    const times = [];

    for (let i = 0; i < numFrames; i++) {
      const centerSample = i * hopLength;
      const time = centerSample / this.config.sampleRate;

      // Extract frame
      const startSample = Math.max(0, centerSample - Math.floor(fftLength / 2));
      const frame = new Float32Array(fftLength);

      for (let j = 0; j < fftLength; j++) {
        const idx = startSample + j;
        frame[j] = idx < audioLength ? audioData[idx] : 0;
      }

      // Copy frame to WASM and compute CQT
      this._copyToWasm(frame, this.audioPtr);
      this.wasm.computeFrameCQT(this.audioPtr, fftLength, this.framePtr);

      // Copy result back
      const frameCQT = this._copyFromWasm(this.framePtr, numBins);
      magnitudes.push(frameCQT);
      times.push(time);
    }

    return {
      magnitudes,
      times,
      numFrames,
      numBins,
      hopLength,
      sampleRate: this.config.sampleRate,
      frequencies: this._getFrequencies(),
      fftLength
    };
  }

  /**
   * Get center frequencies for all bins
   */
  _getFrequencies() {
    const frequencies = new Float32Array(this.config.nBins);
    for (let k = 0; k < this.config.nBins; k++) {
      frequencies[k] = this.wasm.getFrequency(k);
    }
    return frequencies;
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
   * Print features for verification (matching Python output format)
   */
  _printFeatures(features, numBins, targetFrames) {
    const frequencies = this._getFrequencies();

    console.log('\n========== CQT FEATURE EXTRACTION RESULTS (WASM) ==========');
    console.log(`Shape: [${numBins} bins × ${targetFrames} frames] = ${features.length} total values`);

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

    // Print frequency-to-note mapping
    console.log(`\nFrequency bins (first 12):`);
    for (let k = 0; k < Math.min(12, numBins); k++) {
      console.log(`  Bin ${k}: ${frequencies[k].toFixed(2)} Hz (${this._freqToNote(frequencies[k])})`);
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
      const note = this._freqToNote(frequencies[b]);
      let row = `B${b.toString().padStart(2, '0')}(${note.padEnd(3)})|`;
      for (let t = 0; t < previewFrames; t++) {
        const idx = b * targetFrames + t;
        const val = features[idx];
        row += ` ${val.toFixed(2)} `;
      }
      console.log(row);
    }

    console.log('============================================================\n');
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

// Default export
export default CQTExtractor;
