/**
 * Onset Detection Module
 * Implements spectral flux-based onset detection
 */

export class OnsetDetector {
    constructor() {
        this.fftSize = 2048;
    }
    
    /**
     * Detect onsets using spectral flux
     * @param {AudioBuffer} audioBuffer - The audio buffer to analyze
     * @param {Object} config - Configuration parameters
     * @returns {Array} Array of onset objects with time and strength
     */
    detect(audioBuffer, config) {
        const audioData = audioBuffer.getChannelData(0); // Mono or left channel
        const sampleRate = audioBuffer.sampleRate;
        const hopSize = config.audio.hopSize;
        const frameSize = config.onset.frameSize || this.fftSize;
        const threshold = config.onset.threshold;
        const minIntervalMs = config.onset.minInterval;
        const minIntervalSamples = (minIntervalMs / 1000) * sampleRate;
        const smoothingWindow = config.onset.smoothingWindow || 5;
        
        // Calculate spectral flux
        const flux = this.calculateSpectralFlux(audioData, frameSize, hopSize);
        
        // Smooth the flux
        const smoothedFlux = this.smoothSignal(flux, smoothingWindow);
        
        // Normalize flux
        const normalizedFlux = this.normalizeSignal(smoothedFlux);
        
        // Adaptive thresholding
        const adaptiveThreshold = this.calculateAdaptiveThreshold(normalizedFlux, 10, threshold);
        
        // Peak picking
        const peaks = this.pickPeaks(normalizedFlux, adaptiveThreshold, minIntervalSamples / hopSize);
        
        // Convert peak indices to timestamps
        const onsets = peaks.map(peak => ({
            time: (peak.index * hopSize) / sampleRate,
            strength: peak.value,
            sample: peak.index * hopSize
        }));
        
        return onsets;
    }
    
    /**
     * Calculate spectral flux from audio data
     * Spectral flux measures the change in magnitude spectrum between frames
     */
    calculateSpectralFlux(audioData, frameSize, hopSize) {
        const numFrames = Math.floor((audioData.length - frameSize) / hopSize) + 1;
        const flux = new Float32Array(numFrames);
        
        let prevSpectrum = null;
        
        for (let i = 0; i < numFrames; i++) {
            const startSample = i * hopSize;
            const frame = this.extractFrame(audioData, startSample, frameSize);
            
            // Apply Hanning window
            this.applyWindow(frame);
            
            // Compute magnitude spectrum using FFT
            const spectrum = this.computeMagnitudeSpectrum(frame);
            
            if (prevSpectrum) {
                // Calculate spectral flux (half-wave rectified difference)
                let fluxValue = 0;
                for (let j = 0; j < spectrum.length; j++) {
                    const diff = spectrum[j] - prevSpectrum[j];
                    if (diff > 0) {
                        fluxValue += diff * diff; // Squared difference for emphasis
                    }
                }
                flux[i] = Math.sqrt(fluxValue);
            } else {
                flux[i] = 0;
            }
            
            prevSpectrum = spectrum;
        }
        
        return flux;
    }
    
    /**
     * Extract a frame from audio data
     */
    extractFrame(audioData, start, frameSize) {
        const frame = new Float32Array(frameSize);
        for (let i = 0; i < frameSize; i++) {
            if (start + i < audioData.length) {
                frame[i] = audioData[start + i];
            }
        }
        return frame;
    }
    
    /**
     * Apply Hanning window to frame
     */
    applyWindow(frame) {
        const N = frame.length;
        for (let i = 0; i < N; i++) {
            const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
            frame[i] *= window;
        }
    }
    
    /**
     * Compute magnitude spectrum using simple DFT
     * (For production, you'd want to use a proper FFT library)
     */
    computeMagnitudeSpectrum(frame) {
        const N = frame.length;
        const halfN = Math.floor(N / 2);
        const spectrum = new Float32Array(halfN);
        
        // Use a simplified FFT approach (Cooley-Tukey would be better for production)
        // For now, we'll use a basic DFT for lower frequencies only
        const numBins = Math.min(halfN, 256); // Limit bins for performance
        
        for (let k = 0; k < numBins; k++) {
            let real = 0;
            let imag = 0;
            
            for (let n = 0; n < N; n++) {
                const angle = (2 * Math.PI * k * n) / N;
                real += frame[n] * Math.cos(angle);
                imag -= frame[n] * Math.sin(angle);
            }
            
            spectrum[k] = Math.sqrt(real * real + imag * imag) / N;
        }
        
        return spectrum;
    }
    
    /**
     * Smooth signal using moving average
     */
    smoothSignal(signal, windowSize) {
        const smoothed = new Float32Array(signal.length);
        const halfWindow = Math.floor(windowSize / 2);
        
        for (let i = 0; i < signal.length; i++) {
            let sum = 0;
            let count = 0;
            
            for (let j = -halfWindow; j <= halfWindow; j++) {
                const idx = i + j;
                if (idx >= 0 && idx < signal.length) {
                    sum += signal[idx];
                    count++;
                }
            }
            
            smoothed[i] = sum / count;
        }
        
        return smoothed;
    }
    
    /**
     * Normalize signal to 0-1 range
     */
    normalizeSignal(signal) {
        let max = 0;
        for (let i = 0; i < signal.length; i++) {
            if (signal[i] > max) max = signal[i];
        }
        
        if (max === 0) return signal;
        
        const normalized = new Float32Array(signal.length);
        for (let i = 0; i < signal.length; i++) {
            normalized[i] = signal[i] / max;
        }
        
        return normalized;
    }
    
    /**
     * Calculate adaptive threshold
     */
    calculateAdaptiveThreshold(signal, windowSize, baseThreshold) {
        const threshold = new Float32Array(signal.length);
        
        for (let i = 0; i < signal.length; i++) {
            // Calculate local median
            const start = Math.max(0, i - windowSize);
            const end = Math.min(signal.length, i + windowSize + 1);
            const localValues = [];
            
            for (let j = start; j < end; j++) {
                localValues.push(signal[j]);
            }
            
            localValues.sort((a, b) => a - b);
            const median = localValues[Math.floor(localValues.length / 2)];
            
            // Adaptive threshold: base + local median
            threshold[i] = baseThreshold + 0.5 * median;
        }
        
        return threshold;
    }
    
    /**
     * Pick peaks from signal that exceed threshold
     */
    pickPeaks(signal, threshold, minDistance) {
        const peaks = [];
        let lastPeakIndex = -minDistance;
        
        for (let i = 1; i < signal.length - 1; i++) {
            // Check if local maximum
            if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
                // Check if above threshold
                if (signal[i] > threshold[i]) {
                    // Check minimum distance from last peak
                    if (i - lastPeakIndex >= minDistance) {
                        peaks.push({
                            index: i,
                            value: signal[i]
                        });
                        lastPeakIndex = i;
                    } else if (signal[i] > peaks[peaks.length - 1]?.value) {
                        // Replace last peak if this one is stronger
                        peaks[peaks.length - 1] = {
                            index: i,
                            value: signal[i]
                        };
                        lastPeakIndex = i;
                    }
                }
            }
        }
        
        return peaks;
    }
}
