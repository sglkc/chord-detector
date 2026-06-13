/**
 * WCSR (Weighted Chord Symbol Recall) Validator Module
 * Implements MIREX chord recognition evaluation metrics
 */

import { CONFIG } from './config.js';

export class WCSRValidator {
    constructor() {
        // Tolerance for time matching (in seconds)
        this.timeTolerance = 0.05;
    }
    
    /**
     * Calculate WCSR and other metrics
     * @param {Array} predictions - Array of predicted chords with timestamps
     * @param {Array} annotations - Array of ground truth annotations
     * @returns {Object} Validation results
     */
    calculate(predictions, annotations) {
        // Create comparison segments
        const comparisons = this.createComparisons(predictions, annotations);
        
        // Calculate WCSR
        let totalDuration = 0;
        let correctDuration = 0;
        
        for (const annotation of annotations) {
            const duration = annotation.end - annotation.start;
            totalDuration += duration;
            
            // Find overlapping predictions
            const correctTime = this.calculateCorrectTime(annotation, predictions);
            correctDuration += correctTime;
        }
        
        const wcsr = totalDuration > 0 ? correctDuration / totalDuration : 0;
        
        // Calculate per-chord statistics
        const perChordStats = this.calculatePerChordStats(comparisons, annotations);
        
        // Calculate confusion matrix (top errors)
        const confusions = this.calculateConfusions(comparisons);
        
        return {
            wcsr,
            totalDuration,
            correctDuration,
            comparisons,
            perChordStats,
            confusions,
            numAnnotations: annotations.length,
            numPredictions: predictions.length
        };
    }
    
    /**
     * Create segment-by-segment comparisons
     */
    createComparisons(predictions, annotations) {
        const comparisons = [];
        
        for (const annotation of annotations) {
            // Find best matching prediction for this annotation
            let bestPrediction = null;
            let bestOverlap = 0;
            
            for (const prediction of predictions) {
                const overlap = this.calculateOverlap(annotation, prediction);
                if (overlap > bestOverlap) {
                    bestOverlap = overlap;
                    bestPrediction = prediction;
                }
            }
            
            const isMatch = bestPrediction && 
                CONFIG.chords.areEqual(annotation.chord, bestPrediction.mirexChord);
            
            comparisons.push({
                start: annotation.start,
                end: annotation.end,
                groundTruth: annotation.chord,
                predicted: bestPrediction ? bestPrediction.mirexChord : 'N',
                predictedRaw: bestPrediction ? bestPrediction.chord : null,
                confidence: bestPrediction ? bestPrediction.confidence : 0,
                overlap: bestOverlap,
                match: isMatch,
                duration: annotation.end - annotation.start
            });
        }
        
        return comparisons;
    }
    
    /**
     * Calculate overlap between two time segments
     */
    calculateOverlap(segment1, segment2) {
        const overlapStart = Math.max(segment1.start, segment2.start);
        const overlapEnd = Math.min(segment1.end, segment2.end);
        const overlap = Math.max(0, overlapEnd - overlapStart);
        return overlap;
    }
    
    /**
     * Calculate correct time for an annotation
     */
    calculateCorrectTime(annotation, predictions) {
        let correctTime = 0;
        
        for (const prediction of predictions) {
            // Check if chords match
            if (CONFIG.chords.areEqual(annotation.chord, prediction.mirexChord)) {
                // Calculate overlap
                const overlap = this.calculateOverlap(annotation, prediction);
                correctTime += overlap;
            }
        }
        
        // Don't exceed annotation duration
        return Math.min(correctTime, annotation.end - annotation.start);
    }
    
    /**
     * Calculate per-chord statistics
     */
    calculatePerChordStats(comparisons, annotations) {
        const stats = {};
        
        // Initialize stats for each chord in annotations
        for (const annotation of annotations) {
            const chord = annotation.chord;
            if (!stats[chord]) {
                stats[chord] = {
                    totalDuration: 0,
                    correctDuration: 0,
                    count: 0,
                    correctCount: 0
                };
            }
        }
        
        // Calculate stats from comparisons
        for (const comp of comparisons) {
            const chord = comp.groundTruth;
            if (stats[chord]) {
                stats[chord].totalDuration += comp.duration;
                stats[chord].count++;
                
                if (comp.match) {
                    stats[chord].correctDuration += comp.duration;
                    stats[chord].correctCount++;
                }
            }
        }
        
        // Calculate accuracy for each chord
        const result = {};
        for (const [chord, data] of Object.entries(stats)) {
            result[chord] = {
                accuracy: data.totalDuration > 0 ? data.correctDuration / data.totalDuration : 0,
                totalDuration: data.totalDuration,
                correctDuration: data.correctDuration,
                count: data.count,
                correctCount: data.correctCount
            };
        }
        
        return result;
    }
    
    /**
     * Calculate confusion pairs (most common errors)
     */
    calculateConfusions(comparisons) {
        const confusionMap = {};
        
        for (const comp of comparisons) {
            if (!comp.match && comp.predicted !== 'N') {
                const key = `${comp.groundTruth}→${comp.predicted}`;
                if (!confusionMap[key]) {
                    confusionMap[key] = {
                        groundTruth: comp.groundTruth,
                        predicted: comp.predicted,
                        count: 0,
                        totalDuration: 0
                    };
                }
                confusionMap[key].count++;
                confusionMap[key].totalDuration += comp.duration;
            }
        }
        
        // Sort by count and return top errors
        const confusions = Object.values(confusionMap)
            .sort((a, b) => b.count - a.count)
            .slice(0, 10);
        
        return confusions;
    }
    
    /**
     * Calculate frame-level accuracy (alternative metric)
     */
    calculateFrameAccuracy(predictions, annotations, frameSize = 0.1) {
        const duration = Math.max(
            ...annotations.map(a => a.end),
            ...predictions.map(p => p.end)
        );
        
        const numFrames = Math.ceil(duration / frameSize);
        let correctFrames = 0;
        
        for (let i = 0; i < numFrames; i++) {
            const time = i * frameSize;
            
            // Find ground truth chord at this time
            const gtChord = this.findChordAtTime(annotations, time);
            const predChord = this.findChordAtTime(predictions, time, 'mirexChord');
            
            if (gtChord && predChord && CONFIG.chords.areEqual(gtChord, predChord)) {
                correctFrames++;
            }
        }
        
        return numFrames > 0 ? correctFrames / numFrames : 0;
    }
    
    /**
     * Find chord at a specific time
     */
    findChordAtTime(segments, time, chordField = 'chord') {
        for (const segment of segments) {
            if (time >= segment.start && time < segment.end) {
                return segment[chordField];
            }
        }
        return null;
    }
    
    /**
     * Generate detailed report
     */
    generateReport(results) {
        let report = '';
        report += '=== CHORD RECOGNITION VALIDATION REPORT ===\n\n';
        
        report += `WCSR Score: ${(results.wcsr * 100).toFixed(2)}%\n`;
        report += `Total Duration: ${results.totalDuration.toFixed(2)}s\n`;
        report += `Correct Duration: ${results.correctDuration.toFixed(2)}s\n`;
        report += `Annotations: ${results.numAnnotations}\n`;
        report += `Predictions: ${results.numPredictions}\n\n`;
        
        report += '--- Per-Chord Accuracy ---\n';
        for (const [chord, stats] of Object.entries(results.perChordStats)) {
            report += `${chord}: ${(stats.accuracy * 100).toFixed(1)}% `;
            report += `(${stats.correctCount}/${stats.count})\n`;
        }
        
        report += '\n--- Top Confusions ---\n';
        for (const conf of results.confusions) {
            report += `${conf.groundTruth} → ${conf.predicted}: ${conf.count} times\n`;
        }
        
        return report;
    }
}
