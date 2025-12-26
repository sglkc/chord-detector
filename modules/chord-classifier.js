/**
 * Chord Classifier Module
 * Uses TensorFlow.js to run CNN model inference
 */

import { CONFIG, CHORD_LABELS } from './config.js';

export class ChordClassifier {
    constructor() {
        this.model = null;
        this.labels = CHORD_LABELS;
    }
    
    /**
     * Load the TensorFlow.js model
     * @param {string} modelPath - Path to model.json
     */
    async loadModel(modelPath) {
        try {
            // Use tf from global scope (loaded via script tag)
            this.model = await tf.loadLayersModel(modelPath);
            console.log('Model loaded successfully');
            console.log('Input shape:', this.model.inputs[0].shape);
            console.log('Output shape:', this.model.outputs[0].shape);
            
            // Warm up the model with a dummy prediction
            const dummyInput = tf.zeros([1, 36, 200, 1]);
            await this.model.predict(dummyInput).data();
            dummyInput.dispose();
            
            console.log('Model warmed up');
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }
    
    /**
     * Predict chord from CQT features
     * @param {Float32Array} features - CQT features (36 x 200)
     * @returns {Object} Prediction result with chord label and confidence
     */
    async predict(features) {
        if (!this.model) {
            throw new Error('Model not loaded');
        }
        
        // Get expected input shape from model
        const inputShape = this.model.inputs[0].shape;
        const numBins = inputShape[1];      // 36
        const numFrames = inputShape[2];    // 200
        
        // Reshape features to match model input: [batch, bins, frames, channels]
        // Input features are in shape [bins * frames]
        const inputTensor = tf.tidy(() => {
            // Create tensor from features
            let tensor = tf.tensor1d(features);
            
            // Reshape to [bins, frames]
            tensor = tensor.reshape([numBins, numFrames]);
            
            // Add batch and channel dimensions: [1, bins, frames, 1]
            tensor = tensor.expandDims(0).expandDims(-1);
            
            return tensor;
        });
        
        try {
            // Run prediction
            const prediction = await this.model.predict(inputTensor);
            const probabilities = await prediction.data();
            
            // Find the class with highest probability
            let maxProb = 0;
            let maxIndex = 0;
            
            for (let i = 0; i < probabilities.length; i++) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i];
                    maxIndex = i;
                }
            }
            
            // Get chord label
            const chord = this.labels[maxIndex];
            const mirexChord = CONFIG.chords.modelToMirex(chord);
            
            // Get top 3 predictions for debugging
            const topPredictions = this.getTopPredictions(probabilities, 3);
            
            // Clean up
            prediction.dispose();
            inputTensor.dispose();
            
            return {
                chord: chord,
                mirexChord: mirexChord,
                confidence: maxProb,
                classIndex: maxIndex,
                topPredictions: topPredictions,
                allProbabilities: Array.from(probabilities)
            };
        } catch (error) {
            inputTensor.dispose();
            throw error;
        }
    }
    
    /**
     * Get top N predictions
     */
    getTopPredictions(probabilities, n) {
        const indexed = Array.from(probabilities).map((prob, idx) => ({
            index: idx,
            probability: prob,
            chord: this.labels[idx],
            mirexChord: CONFIG.chords.modelToMirex(this.labels[idx])
        }));
        
        indexed.sort((a, b) => b.probability - a.probability);
        
        return indexed.slice(0, n);
    }
    
    /**
     * Batch prediction for multiple windows
     * @param {Array<Float32Array>} featuresList - Array of feature arrays
     * @returns {Array<Object>} Array of prediction results
     */
    async predictBatch(featuresList) {
        if (!this.model) {
            throw new Error('Model not loaded');
        }
        
        const results = [];
        
        for (const features of featuresList) {
            const result = await this.predict(features);
            results.push(result);
        }
        
        return results;
    }
    
    /**
     * Get all chord labels
     */
    getLabels() {
        return this.labels;
    }
    
    /**
     * Get model info
     */
    getModelInfo() {
        if (!this.model) {
            return null;
        }
        
        return {
            inputShape: this.model.inputs[0].shape,
            outputShape: this.model.outputs[0].shape,
            numClasses: this.labels.length,
            labels: this.labels
        };
    }
}
