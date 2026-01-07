/**
 * Configuration Module
 * All configurable parameters for the chord detection system
 */

export const CONFIG = {
    // Audio processing parameters
    audio: {
        sampleRate: 48000,          // Sample rate in Hz
        hopSize: 512,               // Hop size in samples
        minFrequency: 130.8,        // Minimum frequency (C3 = 130.81 Hz)
    },

    // Onset detection parameters
    onset: {
        threshold: 0.15,            // Spectral flux threshold (0.0 - 1.0)
        minInterval: 100,           // Minimum interval between onsets in ms
        preBuffer: 50,              // Pre-onset buffer in ms
        frameSize: 2048,            // FFT frame size for spectral analysis
        smoothingWindow: 5,         // Smoothing window size for flux
        ignoreSubsequentOnsets: false, // If true, only keep first onset per window duration
    },

    // Classification parameters
    classification: {
        windowSize: 2.0,            // Window size in seconds
        cqtBins: 36,                // Number of CQT frequency bins
        cqtTimeFrames: 200,         // Number of time frames for model input
        confidenceThreshold: 0.5,   // Minimum confidence for valid prediction
    },

    // Chord mappings
    chords: {
        // 12 root notes
        roots: ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'],

        // 3 chord types
        types: ['major', 'minor', 'diminished'],

        // Model output format: Root_type_octave (e.g., "C_major_4")
        // MIREX format: Root:type (e.g., "C:maj", "C:min", "C:dim")

        // IMPORTANT: Labels must be in ALPHABETICAL order to match LabelEncoder!
        // LabelEncoder sorts alphabetically, so A# comes BEFORE A in the array.
        // This is the order the model outputs predictions.
        modelLabels: [
            'A#_diminished_4', 'A#_major_4', 'A#_minor_4',
            'A_diminished_4', 'A_major_4', 'A_minor_4',
            'B_diminished_4', 'B_major_4', 'B_minor_4',
            'C#_diminished_4', 'C#_major_4', 'C#_minor_4',
            'C_diminished_4', 'C_major_4', 'C_minor_4',
            'D#_diminished_4', 'D#_major_4', 'D#_minor_4',
            'D_diminished_4', 'D_major_4', 'D_minor_4',
            'E_diminished_4', 'E_major_4', 'E_minor_4',
            'F#_diminished_4', 'F#_major_4', 'F#_minor_4',
            'F_diminished_4', 'F_major_4', 'F_minor_4',
            'G#_diminished_4', 'G#_major_4', 'G#_minor_4',
            'G_diminished_4', 'G_major_4', 'G_minor_4',
        ],

        // Get model labels (returns the pre-defined alphabetical array)
        getModelLabels: function () {
            return this.modelLabels;
        },

        // Convert model label to MIREX format
        modelToMirex: function (modelLabel) {
            const parts = modelLabel.split('_');
            if (parts.length < 2) return modelLabel;

            const root = parts[0];
            const type = parts[1];

            const typeMap = {
                'major': 'maj',
                'minor': 'min',
                'diminished': 'dim'
            };

            return `${root}:${typeMap[type] || type}`;
        },

        // Convert MIREX format to model label
        mirexToModel: function (mirexLabel) {
            const parts = mirexLabel.split(':');
            if (parts.length < 2) return mirexLabel;

            const root = parts[0];
            const type = parts[1];

            const typeMap = {
                'maj': 'major',
                'min': 'minor',
                'dim': 'diminished'
            };

            return `${root}_${typeMap[type] || type}_4`;
        },

        // Normalize chord label for comparison
        normalizeChord: function (chord) {
            // Remove octave info if present
            let normalized = chord.replace(/_\d$/, '');

            // Convert to lowercase for comparison
            normalized = normalized.toLowerCase();

            // Standardize type names
            normalized = normalized.replace(':maj', '_major');
            normalized = normalized.replace(':min', '_minor');
            normalized = normalized.replace(':dim', '_diminished');

            return normalized;
        },

        // Check if two chords are equivalent
        areEqual: function (chord1, chord2) {
            return this.normalizeChord(chord1) === this.normalizeChord(chord2);
        }
    },

    // Visualization parameters
    visualization: {
        cqtColormap: 'viridis',     // Color map for CQT display
        onsetColor: '#ff4757',      // Color for onset markers
        gtColor: '#2ed573',         // Color for ground truth
        predColor: '#4a90d9',       // Color for predictions
    }
};

// Export chord labels for easy access
export const CHORD_LABELS = CONFIG.chords.getModelLabels();
