/**
 * CQT Feature Extraction Module (Pluggable)
 * 
 * This module provides a unified interface for CQT extraction with
 * pluggable backends. Switch between implementations easily.
 * 
 * Available backends:
 * - 'wasm': WebAssembly CQT (fastest, recommended)
 * - 'librosa': Pure JS librosa-compatible CQT (fallback)
 * - 'showcqt': ShowCQT-based visualization CQT (faster, but less accurate for ML)
 * 
 * Usage:
 *   import { CQTExtractor } from './cqt-extractor.js';
 *   
 *   // Use default (wasm)
 *   const extractor = new CQTExtractor();
 *   
 *   // Or specify backend
 *   const extractor = new CQTExtractor('librosa');
 *   
 *   // Switch backend at runtime
 *   extractor.setBackend('wasm');
 */

import { CQTExtractor as WasmCQT } from './cqt-wasm.js';
import { CQTExtractor as LibrosaCQT } from './cqt-librosa.js';
import { ShowCQTExtractor } from './cqt-showcqt.js';

// Available backends
const BACKENDS = {
    'wasm': WasmCQT,
    'librosa': LibrosaCQT,
    'showcqt': ShowCQTExtractor,
};

// Default backend (WASM for best performance)
const DEFAULT_BACKEND = 'wasm';

export class CQTExtractor {
    /**
     * Create a pluggable CQT extractor
     * @param {string} backend - Backend to use: 'librosa' or 'showcqt'
     */
    constructor(backend = DEFAULT_BACKEND) {
        this.backendName = null;
        this.extractor = null;
        this.initialized = false;
        this.sampleRate = 48000;

        this.setBackend(backend);
    }

    /**
     * Set the CQT extraction backend
     * @param {string} backend - Backend name: 'librosa' or 'showcqt'
     */
    setBackend(backend) {
        const backendLower = backend.toLowerCase();

        if (!BACKENDS[backendLower]) {
            console.warn(`Unknown backend '${backend}', using default '${DEFAULT_BACKEND}'`);
            this.backendName = DEFAULT_BACKEND;
        } else {
            this.backendName = backendLower;
        }

        const BackendClass = BACKENDS[this.backendName];
        this.extractor = new BackendClass();
        this.initialized = false;

        console.log(`CQT Extractor backend set to: ${this.backendName}`);
    }

    /**
     * Get the current backend name
     * @returns {string} Current backend name
     */
    getBackend() {
        return this.backendName;
    }

    /**
     * Get list of available backends
     * @returns {string[]} Array of available backend names
     */
    static getAvailableBackends() {
        return Object.keys(BACKENDS);
    }

    /**
     * Initialize the CQT extractor
     * @param {number} sampleRate - Audio sample rate
     */
    async init(sampleRate = 48000) {
        this.sampleRate = sampleRate;
        await this.extractor.init(sampleRate);
        this.initialized = true;
    }

    /**
     * Extract full CQT spectrogram from audio buffer for visualization
     * @param {AudioBuffer} audioBuffer - The audio buffer
     * @param {Object} config - Configuration parameters
     * @returns {Object} CQT data with magnitudes and time info
     */
    async extractFullCQT(audioBuffer, config) {
        if (!this.initialized) {
            await this.init(audioBuffer.sampleRate);
        }

        // Extract audio data from AudioBuffer
        const audioData = audioBuffer.getChannelData(0);

        // Initialize extractor with proper parameters for librosa/wasm backends
        // Only init if not already initialized (avoid repeated WASM memory allocations)
        if ((this.backendName === 'librosa' || this.backendName === 'wasm') && !this.extractor.initialized) {
            await this.extractor.init({
                sampleRate: audioBuffer.sampleRate,
                fmin: config.audio.minFrequency || 130.81,
                nBins: config.classification.cqtBins,
                binsPerOctave: 12,
                hopLength: config.audio.hopSize
            });
        }

        // For librosa/wasm backend, pass Float32Array directly
        // For showcqt backend, pass AudioBuffer (it handles internally)
        let result;
        if (this.backendName === 'librosa' || this.backendName === 'wasm') {
            result = this.extractor.extractFullCQT(audioData, {
                hopLength: config.audio.hopSize
            });
        } else {
            // ShowCQT backend expects AudioBuffer  
            result = await this.extractor.extractFullCQT(audioBuffer, config);
        }

        // Ensure consistent return format
        return {
            magnitudes: result.magnitudes,
            times: result.times,
            numFrames: result.numFrames,
            numBins: result.numBins,
            hopSize: config.audio.hopSize,
            sampleRate: audioBuffer.sampleRate,
            duration: audioBuffer.duration,
            fftLength: result.fftLength,
            timeOffset: result.fftLength ? result.fftLength / (2 * audioBuffer.sampleRate) : 0
        };
    }

    /**
     * Extract CQT features for a single window (for classification)
     * @param {Float32Array} windowData - Audio window data
     * @param {Object} config - Configuration parameters
     * @returns {Float32Array} CQT features shaped for model input
     */
    async extractFeatures(windowData, config) {
        if (!this.initialized) {
            await this.init(config.audio.sampleRate);
        }
        return this.extractor.extractFeatures(windowData, config);
    }

    /**
     * Get CQT visualization as ImageData
     */
    getVisualizationData(magnitudes, numBins, numFrames) {
        return this.extractor.getVisualizationData(magnitudes, numBins, numFrames);
    }

    /**
     * Viridis colormap approximation (for backward compatibility)
     */
    viridisColor(value) {
        if (this.extractor.viridisColor) {
            return this.extractor.viridisColor(value);
        } else if (this.extractor._viridisColor) {
            return this.extractor._viridisColor(value);
        }
        // Fallback
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

// Also export individual backends for direct use
export { WasmCQT, LibrosaCQT, ShowCQTExtractor };

// Export default backend constant
export const CQT_DEFAULT_BACKEND = DEFAULT_BACKEND;
