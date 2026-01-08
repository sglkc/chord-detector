/**
 * Piano Chord Detection Validation System
 * Main Application Module
 */

import { OnsetDetector } from './modules/onset-detector.js';
import { CQTExtractor } from './modules/cqt-extractor.js';
import { ChordClassifier } from './modules/chord-classifier.js';
import { WCSRValidator } from './modules/wcsr-validator.js';
import { Visualizer } from './modules/visualizer.js';
import { CONFIG } from './modules/config.js';

class ChordValidationApp {
    constructor() {
        this.audioFile = null;
        this.annotationFile = null;
        this.audioBuffer = null;
        this.annotations = null;
        this.model = null;

        // Playback state
        this.audioContext = null;
        this.audioSource = null;
        this.isPlaying = false;
        this.playbackStartTime = 0;
        this.playbackOffset = 0;
        this.animationFrameId = null;

        // Current model and backend
        this.currentModelName = CONFIG.classification.model;
        this.currentCqtBackend = CONFIG.classification.cqtBackend;

        this.onsetDetector = new OnsetDetector();
        this.cqtExtractor = new CQTExtractor(this.currentCqtBackend);
        this.classifier = new ChordClassifier();
        this.validator = new WCSRValidator();
        this.visualizer = new Visualizer();

        this.initElements();
        this.bindEvents();
        this.loadModel();
    }

    initElements() {
        // Config inputs
        this.configInputs = {
            sampleRate: document.getElementById('sampleRate'),
            hopSize: document.getElementById('hopSize'),
            minFrequency: document.getElementById('minFrequency'),
            onsetThreshold: document.getElementById('onsetThreshold'),
            minOnsetInterval: document.getElementById('minOnsetInterval'),
            preOnsetBuffer: document.getElementById('preOnsetBuffer'),
            ignoreSubsequentOnsets: document.getElementById('ignoreSubsequentOnsets'),
            windowSize: document.getElementById('windowSize'),
            cqtBins: document.getElementById('cqtBins'),
            confidenceThreshold: document.getElementById('confidenceThreshold'),
            modelSelect: document.getElementById('modelSelect'),
            cqtBackendSelect: document.getElementById('cqtBackendSelect')
        };

        // File inputs
        this.audioFileInput = document.getElementById('audioFile');
        this.annotationFileInput = document.getElementById('annotationFile');
        this.audioFileName = document.getElementById('audioFileName');
        this.annotationFileName = document.getElementById('annotationFileName');

        // Buttons
        this.runButton = document.getElementById('runValidation');

        // Sections
        this.progressSection = document.getElementById('progressSection');
        this.visualizationSection = document.getElementById('visualizationSection');
        this.resultsSection = document.getElementById('resultsSection');

        // Progress
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');

        // Canvas
        this.cqtCanvas = document.getElementById('cqtCanvas');

        // Playback elements
        this.playButton = document.getElementById('playButton');
        this.playbackCursor = document.getElementById('playbackCursor');
        this.playbackTimeDisplay = document.getElementById('playbackTime');
        this.cqtCanvasWrapper = this.cqtCanvas?.parentElement;
    }

    bindEvents() {
        // File input handlers
        this.audioFileInput.addEventListener('change', (e) => this.handleAudioFile(e));
        this.annotationFileInput.addEventListener('change', (e) => this.handleAnnotationFile(e));

        // Run validation
        this.runButton.addEventListener('click', () => this.runValidation());

        // Config change handlers
        Object.values(this.configInputs).forEach(input => {
            if (input) {
                input.addEventListener('change', () => this.updateConfig());
            }
        });

        // Model selection change - requires model reload
        this.configInputs.modelSelect?.addEventListener('change', () => this.handleModelChange());

        // CQT backend change - requires extractor reinitialization
        this.configInputs.cqtBackendSelect?.addEventListener('change', () => this.handleCqtBackendChange());

        // Playback controls
        this.playButton?.addEventListener('click', () => this.togglePlayback());
        this.cqtCanvas?.addEventListener('click', (e) => this.handleCanvasClick(e));
    }

    updateConfig() {
        CONFIG.audio.sampleRate = parseInt(this.configInputs.sampleRate.value);
        CONFIG.audio.hopSize = parseInt(this.configInputs.hopSize.value);
        CONFIG.audio.minFrequency = parseFloat(this.configInputs.minFrequency.value);

        CONFIG.onset.threshold = parseFloat(this.configInputs.onsetThreshold.value);
        CONFIG.onset.minInterval = parseInt(this.configInputs.minOnsetInterval.value);
        CONFIG.onset.preBuffer = parseInt(this.configInputs.preOnsetBuffer.value);
        CONFIG.onset.ignoreSubsequentOnsets = this.configInputs.ignoreSubsequentOnsets.checked;

        CONFIG.classification.windowSize = parseFloat(this.configInputs.windowSize.value);
        CONFIG.classification.cqtBins = parseInt(this.configInputs.cqtBins.value);
        CONFIG.classification.confidenceThreshold = parseFloat(this.configInputs.confidenceThreshold.value);
    }

    async handleAudioFile(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.audioFile = file;
        this.audioFileName.textContent = file.name;
        this.audioFileName.classList.add('selected');

        this.checkReadyState();
    }

    async handleAnnotationFile(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.annotationFile = file;
        this.annotationFileName.textContent = file.name;
        this.annotationFileName.classList.add('selected');

        // Parse annotation file
        const text = await file.text();
        this.annotations = this.parseAnnotations(text);

        this.checkReadyState();
    }

    parseAnnotations(text) {
        const lines = text.trim().split('\n');
        const annotations = [];

        for (const line of lines) {
            const parts = line.trim().split(/\s+/);
            if (parts.length >= 3) {
                const start = parseFloat(parts[0]);
                const end = parseFloat(parts[1]);
                const chord = parts[2];

                if (!isNaN(start) && !isNaN(end)) {
                    annotations.push({ start, end, chord });
                }
            }
        }

        return annotations;
    }

    checkReadyState() {
        const ready = this.audioFile && this.annotationFile && this.model;
        this.runButton.disabled = !ready;
    }

    async loadModel() {
        this.updateProgress(5, 'Loading TensorFlow.js model...');
        try {
            const modelPath = `./models/${this.currentModelName}/model.json`;
            await this.classifier.loadModel(modelPath);
            this.model = this.classifier.model;
            this.checkReadyState();
            this.updateProgress(0, `Model "${this.currentModelName}" loaded successfully`);
            this.progressSection.style.display = 'none';
            console.log(`Loaded model: ${this.currentModelName}`);
        } catch (error) {
            console.error('Failed to load model:', error);
            this.updateProgress(0, 'Failed to load model: ' + error.message);
        }
    }

    async handleModelChange() {
        const newModel = this.configInputs.modelSelect.value;
        if (newModel !== this.currentModelName) {
            this.currentModelName = newModel;
            this.model = null;
            this.classifier = new ChordClassifier();
            await this.loadModel();
        }
    }

    async handleCqtBackendChange() {
        const newBackend = this.configInputs.cqtBackendSelect.value;
        if (newBackend !== this.currentCqtBackend) {
            this.currentCqtBackend = newBackend;
            this.cqtExtractor = new CQTExtractor(newBackend);
            CONFIG.classification.cqtBackend = newBackend;
            console.log(`CQT backend changed to: ${newBackend}`);
        }
    }

    updateProgress(percent, text) {
        this.progressSection.style.display = 'block';
        this.progressFill.style.width = percent + '%';
        this.progressText.textContent = text;
    }

    async runValidation() {
        try {
            this.updateConfig();
            this.visualizationSection.style.display = 'none';
            this.resultsSection.style.display = 'none';

            // Step 1: Load and decode audio
            this.updateProgress(10, 'Loading audio file...');
            this.audioBuffer = await this.loadAudio(this.audioFile);

            // Step 2: Initialize CQT extractor
            this.updateProgress(20, 'Initializing CQT extractor...');
            await this.cqtExtractor.init(CONFIG.audio.sampleRate);

            // Step 3: Extract full CQT for visualization
            this.updateProgress(30, 'Extracting CQT spectrogram...');
            const fullCQT = await this.cqtExtractor.extractFullCQT(this.audioBuffer, CONFIG);

            // Step 4: Detect onsets
            this.updateProgress(50, 'Detecting onsets with spectral flux...');
            let onsets = this.onsetDetector.detect(this.audioBuffer, CONFIG);

            // Step 4.5: Filter subsequent onsets if enabled
            if (CONFIG.onset.ignoreSubsequentOnsets) {
                onsets = this.filterSubsequentOnsets(onsets, CONFIG.classification.windowSize);
            }

            // Step 5: Visualize CQT and onsets
            this.updateProgress(60, 'Rendering visualization...');
            this.visualizationSection.style.display = 'block';
            this.visualizer.drawCQT(this.cqtCanvas, fullCQT, onsets, this.annotations, this.audioBuffer.duration);
            this.visualizer.drawOnsetList(document.getElementById('onsetList'), onsets);

            // Step 6: Classify chords at each onset
            this.updateProgress(70, 'Classifying chords...');
            const predictions = await this.classifyChords(onsets);

            // Step 7: Calculate WCSR
            this.updateProgress(90, 'Calculating WCSR score...');
            const results = this.validator.calculate(predictions, this.annotations);

            // Step 8: Display results
            this.updateProgress(100, 'Validation complete!');
            this.displayResults(results, predictions);

            setTimeout(() => {
                this.progressSection.style.display = 'none';
            }, 1000);

        } catch (error) {
            console.error('Validation error:', error);
            this.updateProgress(0, 'Error: ' + error.message);
        }
    }

    async loadAudio(file) {
        const arrayBuffer = await file.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: CONFIG.audio.sampleRate
        });
        return await audioContext.decodeAudioData(arrayBuffer);
    }

    async classifyChords(onsets) {
        const predictions = [];
        const audioData = this.audioBuffer.getChannelData(0);
        const sampleRate = this.audioBuffer.sampleRate;
        const windowSamples = Math.floor(CONFIG.classification.windowSize * sampleRate);

        for (let i = 0; i < onsets.length; i++) {
            const onset = onsets[i];
            const startSample = Math.floor(onset.time * sampleRate);
            const endSample = Math.min(startSample + windowSamples, audioData.length);

            // Determine the end time for this chord segment
            let endTime;
            if (i < onsets.length - 1) {
                endTime = onsets[i + 1].time;
            } else {
                endTime = this.audioBuffer.duration;
            }

            // Extract window
            const windowData = audioData.slice(startSample, endSample);

            if (windowData.length < windowSamples * 0.5) {
                continue; // Skip if window too short
            }

            // Extract CQT features for this window
            const cqtFeatures = await this.cqtExtractor.extractFeatures(windowData, CONFIG);

            // Classify
            const result = await this.classifier.predict(cqtFeatures);

            predictions.push({
                start: onset.time,
                end: endTime,
                chord: result.chord,
                confidence: result.confidence,
                mirexChord: result.mirexChord
            });
        }

        return predictions;
    }

    displayResults(results, predictions) {
        this.resultsSection.style.display = 'block';

        // WCSR Score
        document.getElementById('wcsrScore').textContent =
            (results.wcsr * 100).toFixed(1) + '%';
        document.getElementById('totalDuration').textContent =
            results.totalDuration.toFixed(2) + 's';
        document.getElementById('correctDuration').textContent =
            results.correctDuration.toFixed(2) + 's';
        document.getElementById('chordsDetected').textContent =
            predictions.length;

        // Timeline
        this.visualizer.drawTimeline(
            document.getElementById('gtTimeline'),
            document.getElementById('predTimeline'),
            document.getElementById('timelineAxis'),
            this.annotations,
            predictions,
            this.audioBuffer.duration
        );

        // Per-chord accuracy
        this.visualizer.drawChordAccuracy(
            document.getElementById('chordAccuracyGrid'),
            results.perChordStats
        );

        // Comparison table
        this.visualizer.drawComparisonTable(
            document.getElementById('comparisonTableBody'),
            results.comparisons
        );

        // Confusion matrix
        this.visualizer.drawConfusionMatrix(
            document.getElementById('confusionGrid'),
            results.confusions
        );
    }

    /**
     * Filter out subsequent onsets that fall within the window duration of a previous onset.
     * This keeps only the first onset in each window period.
     * @param {Array} onsets - Array of onset objects with time property
     * @param {number} windowSize - Window duration in seconds
     * @returns {Array} Filtered onsets
     */
    filterSubsequentOnsets(onsets, windowSize) {
        if (onsets.length === 0) return onsets;

        const filtered = [];
        let lastKeptTime = -Infinity;

        for (const onset of onsets) {
            // Only keep this onset if it's beyond the window duration from the last kept onset
            if (onset.time >= lastKeptTime + windowSize) {
                filtered.push(onset);
                lastKeptTime = onset.time;
            }
        }

        console.log(`Filtered onsets: ${onsets.length} → ${filtered.length} (window: ${windowSize}s)`);
        return filtered;
    }

    /**
     * Toggle audio playback
     */
    togglePlayback() {
        if (this.isPlaying) {
            this.stopPlayback();
        } else {
            this.startPlayback();
        }
    }

    /**
     * Start audio playback from current offset
     */
    startPlayback() {
        if (!this.audioBuffer) {
            console.warn('No audio buffer available');
            return;
        }

        // Create audio context if needed
        if (!this.audioContext || this.audioContext.state === 'closed') {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        // Resume context if suspended
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }

        // Create and start source
        this.audioSource = this.audioContext.createBufferSource();
        this.audioSource.buffer = this.audioBuffer;
        this.audioSource.connect(this.audioContext.destination);

        // Handle playback end
        this.audioSource.onended = () => {
            if (this.isPlaying) {
                this.stopPlayback();
                this.playbackOffset = 0;
                this.updateCursorPosition(0);
            }
        };

        this.playbackStartTime = this.audioContext.currentTime;
        this.audioSource.start(0, this.playbackOffset);
        this.isPlaying = true;

        // Update UI
        this.playButton.classList.add('playing');
        this.playButton.querySelector('.play-icon').textContent = '⏹';
        this.playButton.querySelector('.play-text').textContent = 'Stop';
        this.playbackCursor.classList.add('visible');

        // Start animation loop
        this.updatePlaybackCursor();
    }

    /**
     * Stop audio playback
     */
    stopPlayback() {
        if (this.audioSource) {
            try {
                this.audioSource.stop();
            } catch (e) {
                // Ignore if already stopped
            }
            this.audioSource.disconnect();
            this.audioSource = null;
        }

        // Save current position
        if (this.isPlaying && this.audioContext) {
            const elapsed = this.audioContext.currentTime - this.playbackStartTime;
            this.playbackOffset = Math.min(this.playbackOffset + elapsed, this.audioBuffer.duration);
        }

        this.isPlaying = false;

        // Cancel animation
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        // Update UI
        this.playButton.classList.remove('playing');
        this.playButton.querySelector('.play-icon').textContent = '▶';
        this.playButton.querySelector('.play-text').textContent = 'Play';
    }

    /**
     * Animation loop for cursor and time display
     */
    updatePlaybackCursor() {
        if (!this.isPlaying || !this.audioBuffer) return;

        const elapsed = this.audioContext.currentTime - this.playbackStartTime;
        const currentTime = this.playbackOffset + elapsed;

        if (currentTime >= this.audioBuffer.duration) {
            this.stopPlayback();
            this.playbackOffset = 0;
            this.updateCursorPosition(0);
            return;
        }

        this.updateCursorPosition(currentTime);
        this.animationFrameId = requestAnimationFrame(() => this.updatePlaybackCursor());
    }

    /**
     * Update cursor position and time display
     * @param {number} time - Current time in seconds
     */
    updateCursorPosition(time) {
        if (!this.audioBuffer || !this.cqtCanvasWrapper) return;

        const duration = this.audioBuffer.duration;
        const percentage = (time / duration) * 100;

        // Update cursor position
        this.playbackCursor.style.left = `${percentage}%`;

        // Update time display with millisecond precision
        this.playbackTimeDisplay.textContent = `${this.formatTime(time)} / ${this.formatTime(duration)}`;
    }

    /**
     * Handle click on canvas for seeking
     * @param {MouseEvent} event - Click event
     */
    handleCanvasClick(event) {
        if (!this.audioBuffer) return;

        const rect = this.cqtCanvas.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const percentage = clickX / rect.width;
        const seekTime = percentage * this.audioBuffer.duration;

        // Clamp to valid range
        this.playbackOffset = Math.max(0, Math.min(seekTime, this.audioBuffer.duration));

        // Show cursor
        this.playbackCursor.classList.add('visible');
        this.updateCursorPosition(this.playbackOffset);

        // If playing, restart from new position
        if (this.isPlaying) {
            this.stopPlayback();
            this.startPlayback();
        }

        console.log(`Seek to: ${this.formatTime(this.playbackOffset)}`);
    }

    /**
     * Format time as M:SS.mmm
     * @param {number} seconds - Time in seconds
     * @returns {string} Formatted time string
     */
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);
        return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ChordValidationApp();
});
