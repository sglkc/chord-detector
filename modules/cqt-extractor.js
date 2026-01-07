/**
 * CQT Feature Extraction Module
 * 
 * This module now uses a librosa-compatible CQT implementation
 * instead of ShowCQT (which was designed for visualization).
 * 
 * The new implementation properly matches Python/librosa's cqt() output
 * for accurate feature extraction in chord classification.
 */

import { CQTExtractor as LibrosaCQT } from './cqt-librosa.js';

export class CQTExtractor {
    constructor() {
        this.cqt = new LibrosaCQT();
        this.initialized = false;
        this.sampleRate = 48000;
    }

    /**
     * Initialize the CQT extractor
     * @param {number} sampleRate - Audio sample rate
     */
    async init(sampleRate = 48000) {
        this.sampleRate = sampleRate;

        try {
            // Initialize the librosa-compatible CQT
            this.cqt.init({
                sampleRate: sampleRate,
                fmin: 130.81,  // C3
                nBins: 36,
                binsPerOctave: 12,
                hopLength: 512
            });
            this.initialized = true;
            console.log('CQT Extractor initialized (librosa-compatible)');
        } catch (error) {
            console.error('Failed to initialize CQT:', error);
            throw error;
        }
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

        // Re-initialize with config parameters if needed
        this.cqt.init({
            sampleRate: audioBuffer.sampleRate,
            fmin: config.audio.minFrequency || 130.81,
            nBins: config.classification.cqtBins,
            binsPerOctave: 12,
            hopLength: config.audio.hopSize
        });

        const audioData = audioBuffer.getChannelData(0);

        // Use librosa-compatible CQT extraction
        const result = this.cqt.extractFullCQT(audioData, {
            hopLength: config.audio.hopSize
        });

        return {
            magnitudes: result.magnitudes,
            times: result.times,
            numFrames: result.numFrames,
            numBins: result.numBins,
            hopSize: config.audio.hopSize,
            sampleRate: this.sampleRate,
            duration: audioBuffer.duration,
            fftLength: result.fftLength,
            // Time offset for visualization alignment
            timeOffset: result.fftLength / (2 * this.sampleRate)
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

        return this.cqt.extractFeatures(windowData, config);
    }

    /**
     * Get CQT for visualization as ImageData
     */
    getVisualizationData(magnitudes, numBins, numFrames) {
        return this.cqt.getVisualizationData(magnitudes, numBins, numFrames);
    }

    /**
     * Viridis colormap approximation (for backward compatibility)
     */
    viridisColor(value) {
        return this.cqt._viridisColor(value);
    }
}
