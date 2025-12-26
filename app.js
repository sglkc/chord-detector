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
        
        this.onsetDetector = new OnsetDetector();
        this.cqtExtractor = new CQTExtractor();
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
            windowSize: document.getElementById('windowSize'),
            cqtBins: document.getElementById('cqtBins'),
            confidenceThreshold: document.getElementById('confidenceThreshold')
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
    }
    
    bindEvents() {
        // File input handlers
        this.audioFileInput.addEventListener('change', (e) => this.handleAudioFile(e));
        this.annotationFileInput.addEventListener('change', (e) => this.handleAnnotationFile(e));
        
        // Run validation
        this.runButton.addEventListener('click', () => this.runValidation());
        
        // Config change handlers
        Object.values(this.configInputs).forEach(input => {
            input.addEventListener('change', () => this.updateConfig());
        });
    }
    
    updateConfig() {
        CONFIG.audio.sampleRate = parseInt(this.configInputs.sampleRate.value);
        CONFIG.audio.hopSize = parseInt(this.configInputs.hopSize.value);
        CONFIG.audio.minFrequency = parseFloat(this.configInputs.minFrequency.value);
        
        CONFIG.onset.threshold = parseFloat(this.configInputs.onsetThreshold.value);
        CONFIG.onset.minInterval = parseInt(this.configInputs.minOnsetInterval.value);
        CONFIG.onset.preBuffer = parseInt(this.configInputs.preOnsetBuffer.value);
        
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
            await this.classifier.loadModel('./model/model.json');
            this.model = this.classifier.model;
            this.checkReadyState();
            this.updateProgress(0, 'Model loaded successfully');
            this.progressSection.style.display = 'none';
        } catch (error) {
            console.error('Failed to load model:', error);
            this.updateProgress(0, 'Failed to load model: ' + error.message);
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
            const onsets = this.onsetDetector.detect(this.audioBuffer, CONFIG);
            
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
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ChordValidationApp();
});
