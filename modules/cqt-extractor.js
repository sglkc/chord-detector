/**
 * CQT Feature Extraction Module
 * Uses showcqt-js for Constant-Q Transform extraction
 */

import ShowCQT from '../showcqt/showcqt-main.mjs';

export class CQTExtractor {
    constructor() {
        this.cqt = null;
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
            // Instantiate ShowCQT
            this.cqt = await ShowCQT.instantiate({ simd: true });
            this.initialized = true;
            console.log('CQT Extractor initialized with ShowCQT v' + ShowCQT.version);
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

        const audioData = audioBuffer.getChannelData(0);
        const hopSize = config.audio.hopSize;
        const width = config.classification.cqtBins;
        const height = 1; // We only need the color/magnitude data

        // Initialize CQT with parameters
        // ShowCQT uses a fixed frequency range (E0 to E10), but we can work with it
        this.cqt.init(this.sampleRate, width, height, 1.0, 1.0, 1);

        const fftSize = this.cqt.fft_size;
        const numFrames = Math.floor((audioData.length - fftSize) / hopSize) + 1;

        // Store magnitude data for each frame
        const magnitudes = [];
        const times = [];

        for (let i = 0; i < numFrames; i++) {
            const startSample = i * hopSize;
            const time = startSample / this.sampleRate;

            // Fill input buffer
            this.fillInputBuffer(audioData, startSample, fftSize);

            // Calculate CQT
            this.cqt.calc();

            // Get color/magnitude data
            const frameData = new Float32Array(width);
            for (let j = 0; j < width; j++) {
                // The color array contains RGBA values, we use the magnitude
                frameData[j] = this.cqt.color[j * 4]; // R channel as magnitude
            }

            magnitudes.push(frameData);
            times.push(time);
        }

        return {
            magnitudes,
            times,
            numFrames,
            numBins: width,
            hopSize,
            sampleRate: this.sampleRate,
            duration: audioBuffer.duration,
            fftSize,
            // Time offset: the spectrogram starts early because FFT frames are centered
            // The first frame represents audio centered at fftSize/2, not at time 0
            timeOffset: fftSize / (2 * this.sampleRate)
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

        const hopSize = config.audio.hopSize;
        const numBins = config.classification.cqtBins;
        const targetFrames = config.classification.cqtTimeFrames;
        const width = numBins;
        const height = 1;

        // Re-initialize with correct width
        this.cqt.init(this.sampleRate, width, height, 1.0, 1.0, 1);

        const fftSize = this.cqt.fft_size;

        // Calculate number of frames we can extract
        const numFrames = Math.max(1, Math.floor((windowData.length - fftSize) / hopSize) + 1);

        // Extract CQT frames
        const frames = [];

        for (let i = 0; i < numFrames && i < targetFrames; i++) {
            const startSample = i * hopSize;

            // Fill input buffer with window data
            this.fillInputBufferFromArray(windowData, startSample, fftSize);

            // Calculate CQT
            this.cqt.calc();

            // Extract magnitude data
            const frameData = new Float32Array(numBins);
            for (let j = 0; j < numBins; j++) {
                frameData[j] = this.cqt.color[j * 4];
            }

            frames.push(frameData);
        }

        // Pad or truncate to target frames
        const features = this.padOrTruncate(frames, numBins, targetFrames);

        // Normalize features
        this.normalizeFeatures(features);

        // Print features for verification
        this.printFeatures(features, numBins, targetFrames, frames.length);

        return features;
    }

    /**
     * Fill the CQT input buffer from audio data
     */
    fillInputBuffer(audioData, startSample, fftSize) {
        const input = this.cqt.inputs[0];

        for (let i = 0; i < fftSize; i++) {
            const sampleIndex = startSample + i;
            if (sampleIndex < audioData.length) {
                input[i] = audioData[sampleIndex];
            } else {
                input[i] = 0;
            }
        }

        // Fill second channel with same data (mono)
        const input2 = this.cqt.inputs[1];
        for (let i = 0; i < fftSize; i++) {
            input2[i] = input[i];
        }
    }

    /**
     * Fill input buffer from a Float32Array
     */
    fillInputBufferFromArray(data, startSample, fftSize) {
        const input = this.cqt.inputs[0];
        const input2 = this.cqt.inputs[1];

        for (let i = 0; i < fftSize; i++) {
            const sampleIndex = startSample + i;
            const value = sampleIndex < data.length ? data[sampleIndex] : 0;
            input[i] = value;
            input2[i] = value;
        }
    }

    /**
     * Pad or truncate frames to target size
     */
    padOrTruncate(frames, numBins, targetFrames) {
        const features = new Float32Array(numBins * targetFrames);

        for (let t = 0; t < targetFrames; t++) {
            for (let b = 0; b < numBins; b++) {
                if (t < frames.length) {
                    features[b * targetFrames + t] = frames[t][b];
                } else {
                    // Pad with zeros or repeat last frame
                    if (frames.length > 0) {
                        features[b * targetFrames + t] = frames[frames.length - 1][b];
                    } else {
                        features[b * targetFrames + t] = 0;
                    }
                }
            }
        }

        return features;
    }

    /**
     * Print features for verification
     * @param {Float32Array} features - The extracted features
     * @param {number} numBins - Number of frequency bins
     * @param {number} targetFrames - Number of time frames
     * @param {number} actualFrames - Actual frames extracted before padding
     */
    printFeatures(features, numBins, targetFrames, actualFrames) {
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

        // Reshape to 2D for visualization [numBins x targetFrames]
        // Note: features are stored as [bin0_time0, bin0_time1, ..., bin1_time0, bin1_time1, ...]
        console.log(`\nFeature Matrix Preview (top 12 bins × first 8 frames):`);
        console.log(`Note: Layout is [bins (rows) × time frames (columns)]`);

        const previewBins = Math.min(12, numBins);
        const previewFrames = Math.min(8, targetFrames);

        // Print header
        let header = 'Bin\\Time |';
        for (let t = 0; t < previewFrames; t++) {
            header += ` T${t.toString().padStart(2, '0')} `;
        }
        console.log(header);
        console.log('-'.repeat(header.length));

        // Print values for each bin (high frequencies at top)
        for (let b = numBins - 1; b >= numBins - previewBins; b--) {
            let row = `B${b.toString().padStart(3, '0')}    |`;
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
            // Only print every 4th row to keep output manageable
            if (b % 4 === 0 || b === numBins - 1) {
                console.log(`B${b.toString().padStart(3, '0')} ${row}`);
            }
        }

        console.log('====================================================\n');
    }

    /**
     * Normalize features to 0-1 range
     */
    normalizeFeatures(features) {
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
     * Get CQT for visualization as ImageData
     */
    getVisualizationData(magnitudes, numBins, numFrames) {
        const imageData = new ImageData(numFrames, numBins);

        for (let t = 0; t < numFrames; t++) {
            for (let b = 0; b < numBins; b++) {
                const value = magnitudes[t][b];
                const pixelIndex = ((numBins - 1 - b) * numFrames + t) * 4;

                // Viridis-like colormap
                const color = this.viridisColor(value);
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
    viridisColor(value) {
        // Clamp value between 0 and 1
        const v = Math.max(0, Math.min(1, value));

        // Viridis approximation
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
