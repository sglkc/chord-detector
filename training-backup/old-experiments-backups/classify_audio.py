#!/usr/bin/env python3
"""
Audio Classification Script

Classifies a chord from an audio file using the trained CNN model.
This script uses the same CQT extraction as the training pipeline
to verify if the JavaScript extractor matches.

Usage:
    python classify_audio.py <audio_file>
    python classify_audio.py datasets/normal/A_major_4/A_major_4-100.wav
"""

import numpy as np
import librosa
import tensorflow as tf
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============== Configuration (matching JavaScript config.js) ==============
SAMPLE_RATE = 48000
HOP_LENGTH = 512
FMIN = librosa.note_to_hz('C3')  # ~130.81 Hz
N_BINS = 36  # 3 octaves × 12 bins
BINS_PER_OCTAVE = 12
MAX_CQT_PAD_LEN = 200

# Chord labels (MUST match the order used during training by LabelEncoder)
# LabelEncoder sorts labels alphabetically, so A# comes before A
# This was the bug - we had wrong label ordering!
CHORD_LABELS = [
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
]


def extract_cqt_features(audio_path: str, verbose: bool = True) -> np.ndarray:
    """
    Extract CQT features from an audio file.
    Matches the training pipeline exactly.
    """
    # Load audio at target sample rate
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    if verbose:
        duration = len(y) / sr
        print(f'\nLoaded: {audio_path}')
        print(f'  Sample rate: {sr} Hz')
        print(f'  Duration: {duration:.2f} seconds')
        print(f'  Samples: {len(y)}')
    
    # Extract CQT
    cqt = librosa.cqt(
        y=y,
        sr=sr,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        hop_length=HOP_LENGTH
    )
    
    # Take magnitude
    cqt_magnitude = np.abs(cqt)
    
    if verbose:
        print(f'  CQT shape (before padding): {cqt_magnitude.shape}')
    
    # Pad or truncate to fixed length
    if cqt_magnitude.shape[1] < MAX_CQT_PAD_LEN:
        pad_width = MAX_CQT_PAD_LEN - cqt_magnitude.shape[1]
        cqt_magnitude = np.pad(cqt_magnitude, 
                               pad_width=((0, 0), (0, pad_width)), 
                               mode='constant')
    else:
        cqt_magnitude = cqt_magnitude[:, :MAX_CQT_PAD_LEN]
    
    # Normalize to 0-1 range
    min_val = cqt_magnitude.min()
    max_val = cqt_magnitude.max()
    if max_val - min_val > 0:
        cqt_magnitude = (cqt_magnitude - min_val) / (max_val - min_val)
    
    if verbose:
        print(f'  CQT shape (after padding): {cqt_magnitude.shape}')
        print(f'  Value range: [{cqt_magnitude.min():.4f}, {cqt_magnitude.max():.4f}]')
        print(f'  Mean: {cqt_magnitude.mean():.4f}')
    
    return cqt_magnitude


def print_feature_comparison(features: np.ndarray):
    """Print feature preview for comparison with JavaScript."""
    n_bins, n_frames = features.shape
    
    print('\n========== CQT FEATURE COMPARISON ==========')
    print(f'Shape: [{n_bins} bins × {n_frames} frames]')
    print(f'Mean: {features.mean():.6f}')
    print(f'Non-zero: {np.sum(features > 0.001)} ({np.sum(features > 0.001) / features.size * 100:.1f}%)')
    
    # Print frequency bins
    print(f'\nFrequency bins:')
    for k in range(min(12, n_bins)):
        freq = FMIN * (2.0 ** (k / BINS_PER_OCTAVE))
        note = librosa.hz_to_note(freq)
        print(f'  Bin {k}: {freq:.2f} Hz ({note})')
    
    # Print first few frames of top bins
    print(f'\nTop 12 bins × first 8 frames:')
    print('Bin\\Time |', end='')
    for t in range(8):
        print(f' T{t:02d} ', end='')
    print()
    print('-' * 50)
    
    for b in range(n_bins - 1, n_bins - 13, -1):
        freq = FMIN * (2.0 ** (b / BINS_PER_OCTAVE))
        note = librosa.hz_to_note(freq)
        print(f'B{b:02d}({note:4s})|', end='')
        for t in range(8):
            print(f' {features[b, t]:.2f} ', end='')
        print()
    
    print('=' * 45 + '\n')


def classify_audio(audio_path: str, model_path: str, verbose: bool = True, top_k: int = 5):
    """
    Classify a chord from an audio file.
    
    Args:
        audio_path: Path to the audio file
        model_path: Path to the trained model (.keras or SavedModel directory)
        verbose: Print detailed output
        top_k: Number of top predictions to show
    """
    # Extract features
    features = extract_cqt_features(audio_path, verbose)
    
    if verbose:
        print_feature_comparison(features)
    
    # Load model
    print(f'Loading model from: {model_path}')
    model = tf.keras.models.load_model(model_path)
    
    if verbose:
        print(f'Model input shape: {model.input_shape}')
        print(f'Model output shape: {model.output_shape}')
    
    # Prepare input (add batch and channel dimensions)
    # Shape: (1, n_bins, n_frames, 1)
    input_data = features[np.newaxis, :, :, np.newaxis]
    
    if verbose:
        print(f'Input data shape: {input_data.shape}')
    
    # Make prediction
    predictions = model.predict(input_data, verbose=0)[0]  # Remove batch dimension
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    
    print('\n' + '=' * 50)
    print('CLASSIFICATION RESULTS')
    print('=' * 50)
    print(f'\nAudio file: {audio_path}')
    print(f'\nTop {top_k} predictions:')
    print('-' * 40)
    
    for i, idx in enumerate(top_indices):
        label = CHORD_LABELS[idx]
        confidence = predictions[idx] * 100
        bar = '█' * int(confidence / 5) + '░' * (20 - int(confidence / 5))
        print(f'{i+1}. {label:20s} {bar} {confidence:6.2f}%')
    
    # Show the predicted chord
    predicted_idx = top_indices[0]
    predicted_label = CHORD_LABELS[predicted_idx]
    predicted_confidence = predictions[predicted_idx] * 100
    
    print('\n' + '-' * 40)
    print(f'🎵 PREDICTED: {predicted_label} ({predicted_confidence:.1f}% confidence)')
    print('=' * 50 + '\n')
    
    # Return prediction info
    return {
        'predicted_label': predicted_label,
        'predicted_index': predicted_idx,
        'confidence': predicted_confidence,
        'all_predictions': predictions,
        'top_k_labels': [CHORD_LABELS[i] for i in top_indices],
        'top_k_confidences': [predictions[i] * 100 for i in top_indices]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Classify a chord from an audio file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classify_audio.py audio.wav
  python classify_audio.py datasets/normal/A_major_4/A_major_4-100.wav
  python classify_audio.py audio.wav --model models/normal-delay-model.keras
        """
    )
    
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('--model', type=str, default='models/normal-delay-model.keras',
                        help='Path to trained model (default: models/normal-delay-model.keras)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top predictions to show (default: 5)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Validate paths
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print("\nAvailable models:")
        for f in Path('models').glob('*.keras'):
            print(f"  {f}")
        for d in Path('models').iterdir():
            if d.is_dir():
                print(f"  {d}/")
        return 1
    
    # Classify
    result = classify_audio(
        audio_path=str(audio_path),
        model_path=str(model_path),
        verbose=not args.quiet,
        top_k=args.top_k
    )
    
    return 0


if __name__ == '__main__':
    exit(main())