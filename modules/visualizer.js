/**
 * Visualization Module
 * Handles all visual representations of data
 */

export class Visualizer {
    constructor() {
        this.colors = {
            onset: '#ff4757',
            groundTruth: 'rgba(46, 213, 115, 0.3)',
            correct: '#27ae60',
            incorrect: '#e74c3c',
            primary: '#4a90d9'
        };
    }
    
    /**
     * Draw CQT spectrogram with onset markers
     * @param {HTMLCanvasElement} canvas - The canvas element
     * @param {Object} cqtData - CQT magnitude data
     * @param {Array} onsets - Detected onsets
     * @param {Array} annotations - Ground truth annotations
     * @param {number} duration - Audio duration
     */
    drawCQT(canvas, cqtData, onsets, annotations, duration) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);
        
        const { magnitudes, numFrames, numBins } = cqtData;
        
        // Calculate scaling
        const pixelsPerFrame = width / numFrames;
        const pixelsPerBin = height / numBins;
        
        // Find min/max for normalization
        let minVal = Infinity, maxVal = -Infinity;
        for (const frame of magnitudes) {
            for (const val of frame) {
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }
        }
        const range = maxVal - minVal || 1;
        
        // Draw spectrogram
        for (let t = 0; t < numFrames; t++) {
            for (let b = 0; b < numBins; b++) {
                const value = (magnitudes[t][b] - minVal) / range;
                const color = this.viridisColor(value);
                
                ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                ctx.fillRect(
                    t * pixelsPerFrame,
                    (numBins - 1 - b) * pixelsPerBin,
                    Math.ceil(pixelsPerFrame),
                    Math.ceil(pixelsPerBin)
                );
            }
        }
        
        // Draw ground truth regions
        ctx.fillStyle = this.colors.groundTruth;
        for (const annotation of annotations) {
            const x = (annotation.start / duration) * width;
            const w = ((annotation.end - annotation.start) / duration) * width;
            ctx.fillRect(x, 0, w, height);
            
            // Draw chord label
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.font = '10px monospace';
            ctx.save();
            ctx.translate(x + 5, height - 5);
            ctx.fillText(annotation.chord, 0, 0);
            ctx.restore();
            ctx.fillStyle = this.colors.groundTruth;
        }
        
        // Draw onset markers
        ctx.strokeStyle = this.colors.onset;
        ctx.lineWidth = 2;
        
        for (const onset of onsets) {
            const x = (onset.time / duration) * width;
            
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
            
            // Draw timestamp
            ctx.fillStyle = this.colors.onset;
            ctx.font = 'bold 9px monospace';
            ctx.fillText(onset.time.toFixed(3) + 's', x + 2, 12);
        }
        
        // Draw time axis
        this.drawTimeAxis(ctx, width, height, duration);
    }
    
    /**
     * Draw time axis on canvas
     */
    drawTimeAxis(ctx, width, height, duration) {
        const numTicks = Math.min(20, Math.ceil(duration));
        const tickInterval = duration / numTicks;
        
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.font = '10px sans-serif';
        
        for (let i = 0; i <= numTicks; i++) {
            const time = i * tickInterval;
            const x = (time / duration) * width;
            
            // Draw tick
            ctx.fillRect(x, height - 15, 1, 5);
            
            // Draw label every 2 ticks
            if (i % 2 === 0) {
                ctx.fillText(time.toFixed(1) + 's', x - 10, height - 3);
            }
        }
    }
    
    /**
     * Draw onset list
     */
    drawOnsetList(container, onsets) {
        container.innerHTML = '';
        
        for (const onset of onsets) {
            const item = document.createElement('span');
            item.className = 'onset-item';
            item.textContent = `${onset.time.toFixed(3)}s`;
            item.title = `Strength: ${onset.strength.toFixed(3)}`;
            container.appendChild(item);
        }
    }
    
    /**
     * Draw chord timelines
     */
    drawTimeline(gtContainer, predContainer, axisContainer, annotations, predictions, duration) {
        // Clear containers
        gtContainer.innerHTML = '';
        predContainer.innerHTML = '';
        axisContainer.innerHTML = '';
        
        // Draw ground truth timeline
        for (const annotation of annotations) {
            const element = this.createTimelineChord(
                annotation.start,
                annotation.end,
                annotation.chord,
                duration,
                'gt'
            );
            gtContainer.appendChild(element);
        }
        
        // Draw predictions timeline
        for (const prediction of predictions) {
            // Check if correct
            const isCorrect = annotations.some(ann => 
                this.overlaps(ann, prediction) && 
                this.chordsMatch(ann.chord, prediction.mirexChord)
            );
            
            const element = this.createTimelineChord(
                prediction.start,
                prediction.end,
                prediction.mirexChord,
                duration,
                isCorrect ? 'correct' : 'incorrect'
            );
            predContainer.appendChild(element);
        }
        
        // Draw time axis
        const numLabels = Math.min(10, Math.ceil(duration));
        for (let i = 0; i <= numLabels; i++) {
            const time = (i / numLabels) * duration;
            const label = document.createElement('span');
            label.textContent = time.toFixed(1) + 's';
            axisContainer.appendChild(label);
        }
    }
    
    /**
     * Create a timeline chord element
     */
    createTimelineChord(start, end, chord, duration, type) {
        const element = document.createElement('div');
        element.className = `timeline-chord ${type}`;
        element.style.left = `${(start / duration) * 100}%`;
        element.style.width = `${((end - start) / duration) * 100}%`;
        element.textContent = chord;
        element.title = `${start.toFixed(2)}s - ${end.toFixed(2)}s: ${chord}`;
        return element;
    }
    
    /**
     * Check if two segments overlap
     */
    overlaps(seg1, seg2) {
        return seg1.start < seg2.end && seg2.start < seg1.end;
    }
    
    /**
     * Check if two chords match (considering different formats)
     */
    chordsMatch(chord1, chord2) {
        const normalize = (c) => c.toLowerCase()
            .replace(':maj', '_major')
            .replace(':min', '_minor')
            .replace(':dim', '_diminished')
            .replace(/_\d$/, '');
        
        return normalize(chord1) === normalize(chord2);
    }
    
    /**
     * Draw per-chord accuracy grid
     */
    drawChordAccuracy(container, perChordStats) {
        container.innerHTML = '';
        
        // Sort by accuracy
        const sorted = Object.entries(perChordStats)
            .sort((a, b) => b[1].accuracy - a[1].accuracy);
        
        for (const [chord, stats] of sorted) {
            const item = document.createElement('div');
            item.className = 'chord-accuracy-item';
            
            const accuracy = stats.accuracy * 100;
            const color = this.getAccuracyColor(accuracy);
            
            item.innerHTML = `
                <div class="chord-name">${chord}</div>
                <div class="chord-stats">${stats.correctCount}/${stats.count} (${accuracy.toFixed(0)}%)</div>
                <div class="accuracy-bar">
                    <div class="accuracy-fill" style="width: ${accuracy}%; background: ${color}"></div>
                </div>
            `;
            
            container.appendChild(item);
        }
    }
    
    /**
     * Get color based on accuracy
     */
    getAccuracyColor(accuracy) {
        if (accuracy >= 80) return '#27ae60';
        if (accuracy >= 60) return '#f39c12';
        if (accuracy >= 40) return '#e67e22';
        return '#e74c3c';
    }
    
    /**
     * Draw comparison table
     */
    drawComparisonTable(tbody, comparisons) {
        tbody.innerHTML = '';
        
        for (const comp of comparisons) {
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td>${comp.start.toFixed(3)}</td>
                <td>${comp.end.toFixed(3)}</td>
                <td>${comp.groundTruth}</td>
                <td>${comp.predicted}</td>
                <td>${(comp.confidence * 100).toFixed(1)}%</td>
                <td class="${comp.match ? 'match-yes' : 'match-no'}">
                    ${comp.match ? '✓' : '✗'}
                </td>
            `;
            
            tbody.appendChild(row);
        }
    }
    
    /**
     * Draw confusion matrix (top errors)
     */
    drawConfusionMatrix(container, confusions) {
        container.innerHTML = '';
        
        if (confusions.length === 0) {
            container.innerHTML = '<p style="color: var(--success-color);">No errors! Perfect recognition.</p>';
            return;
        }
        
        for (const conf of confusions) {
            const item = document.createElement('div');
            item.className = 'confusion-item';
            
            item.innerHTML = `
                <span class="confusion-pair">
                    <strong>${conf.groundTruth}</strong>
                    <span class="arrow">→</span>
                    ${conf.predicted}
                </span>
                <span class="confusion-count">${conf.count}</span>
            `;
            
            container.appendChild(item);
        }
    }
    
    /**
     * Viridis colormap
     */
    viridisColor(value) {
        const v = Math.max(0, Math.min(1, value));
        
        // Viridis approximation
        const r = Math.round(255 * Math.max(0, Math.min(1, 
            0.267004 + v * (0.329415 + v * (-0.508378 + v * 1.137680)))));
        const g = Math.round(255 * Math.max(0, Math.min(1,
            0.004874 + v * (0.873158 + v * (-0.058404 + v * -0.322897)))));
        const b = Math.round(255 * Math.max(0, Math.min(1,
            0.329415 + v * (0.280197 + v * (-1.314181 + v * 1.171356)))));
        
        return [r, g, b];
    }
}
