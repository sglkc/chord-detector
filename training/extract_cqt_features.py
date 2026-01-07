#!/usr/bin/env python3
"""
CQT Feature Extraction Script
Extracts Constant-Q Transform features from audio files.
Designed to match the JavaScript CQT extractor output as closely as possible.
"""

import numpy as np
import librosa
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============== Configuration ==============
# Sample rate: 48 kHz (matching JavaScript)
SAMPLE_RATE = 48000

# Hop size: 512 samples (matching JavaScript config.audio.hopSize)
HOP_LENGTH = 512

# Frequency range: C3 to C6 (3 octaves)
# JavaScript uses showcqt which has a different frequency mapping,
# but we aim to cover the piano chord range
FMIN = librosa.note_to_hz('C3')  # ~130.81 Hz

# Number of bins: 36 (3 octaves × 12 bins per octave)
N_BINS = 12 * 3

# Bins per octave (standard chromatic scale)
BINS_PER_OCTAVE = 12

# Maximum time frames to pad/truncate to
# 48000 Hz * 2 seconds / 512 hop = ~187.5 frames
# Using 200 for safety margin
MAX_CQT_PAD_LEN = 200


def preprocess(cqt: np.ndarray, max_pad_len: int = MAX_CQT_PAD_LEN) -> np.ndarray:
    """
    Pad or truncate CQT to fixed length.
    
    Args:
        cqt: CQT magnitude array of shape [n_bins, time_frames]
        max_pad_len: Target number of time frames
        
    Returns:
        Padded/truncated CQT of shape [n_bins, max_pad_len]
    """
    if cqt.shape[1] < max_pad_len:
        # Pad with zeros (similar to JS padOrTruncate)
        pad_width = max_pad_len - cqt.shape[1]
        cqt = np.pad(cqt, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate to max length
        cqt = cqt[:, :max_pad_len]
    return cqt


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to 0-1 range (matching JavaScript normalizeFeatures).
    
    Args:
        features: Feature array
        
    Returns:
        Normalized feature array
    """
    min_val = features.min()
    max_val = features.max()
    range_val = max_val - min_val
    
    if range_val > 0:
        return (features - min_val) / range_val
    return features


def print_features(features: np.ndarray, actual_frames: int):
    """
    Print features for verification (matching JavaScript printFeatures).
    
    Args:
        features: Normalized CQT features of shape [n_bins, time_frames]
        actual_frames: Number of frames before padding
    """
    n_bins, target_frames = features.shape
    total_values = features.size
    
    print('\n' + '=' * 52)
    print('========== CQT FEATURE EXTRACTION RESULTS ==========')
    print('=' * 52)
    print(f'Shape: [{n_bins} bins × {target_frames} frames] = {total_values} total values')
    print(f'Actual frames extracted: {actual_frames} (padded to {target_frames})')
    
    # Calculate statistics
    min_val = features.min()
    max_val = features.max()
    mean_val = features.mean()
    non_zero_count = np.sum(features > 0.001)
    non_zero_percent = (non_zero_count / total_values) * 100
    
    print(f'\nStatistics (after normalization):')
    print(f'  Min: {min_val:.6f}')
    print(f'  Max: {max_val:.6f}')
    print(f'  Mean: {mean_val:.6f}')
    print(f'  Non-zero values: {non_zero_count} ({non_zero_percent:.1f}%)')
    
    # Feature Matrix Preview
    preview_bins = min(12, n_bins)
    preview_frames = min(8, target_frames)
    
    print(f'\nFeature Matrix Preview (top {preview_bins} bins × first {preview_frames} frames):')
    print('Note: Layout is [bins (rows) × time frames (columns)]')
    
    # Print header
    header = 'Bin\\Time |'
    for t in range(preview_frames):
        header += f' T{t:02d} '
    print(header)
    print('-' * len(header))
    
    # Print values for each bin (high frequencies at top)
    for b in range(n_bins - 1, n_bins - preview_bins - 1, -1):
        row = f'B{b:03d}    |'
        for t in range(preview_frames):
            val = features[b, t]
            row += f' {val:.2f} '
        print(row)
    
    # ASCII heatmap visualization
    print(f'\nASCII Heatmap (all {n_bins} bins × {preview_frames} frames):')
    print('Legend: ░ (0.0-0.2) ▒ (0.2-0.4) ▓ (0.4-0.6) █ (0.6-0.8) ▀ (0.8-1.0)')
    
    heatmap_chars = ['░', '▒', '▓', '█', '▀']
    
    for b in range(n_bins - 1, -1, -1):
        row = ''
        for t in range(preview_frames):
            val = features[b, t]
            char_idx = min(4, int(val * 5))
            row += heatmap_chars[char_idx]
        # Print every 4th row to keep output manageable
        if b % 4 == 0 or b == n_bins - 1:
            print(f'B{b:03d} {row}')
    
    print('=' * 52 + '\n')


def extract_cqt_features(
    audio_path: str,
    fmin: float = FMIN,
    n_bins: int = N_BINS,
    hop_length: int = HOP_LENGTH,
    max_pad_len: int = MAX_CQT_PAD_LEN,
    sample_rate: int = SAMPLE_RATE,
    normalize: bool = True,
    verbose: bool = True
) -> np.ndarray | None:
    """
    Extract CQT features from an audio file.
    
    Args:
        audio_path: Path to the audio file
        fmin: Minimum frequency (Hz)
        n_bins: Number of frequency bins
        hop_length: Hop size in samples
        max_pad_len: Maximum time frames (for padding/truncation)
        sample_rate: Target sample rate (None to use original)
        normalize: Whether to normalize features to 0-1 range
        verbose: Whether to print feature details
        
    Returns:
        CQT features of shape [n_bins, max_pad_len] or None on error
    """
    try:
        # Load audio at target sample rate
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
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
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=BINS_PER_OCTAVE,
            hop_length=hop_length
        )
        
        # Take magnitude (matching JavaScript's approach)
        cqt_magnitude = np.abs(cqt)
        
        actual_frames = cqt_magnitude.shape[1]
        
        if verbose:
            print(f'  CQT shape (before padding): {cqt_magnitude.shape}')
        
        # Pad/truncate to fixed length
        cqt_padded = preprocess(cqt_magnitude, max_pad_len)
        
        # Normalize to 0-1 range (matching JavaScript normalizeFeatures)
        if normalize:
            cqt_normalized = normalize_features(cqt_padded)
        else:
            cqt_normalized = cqt_padded
        
        if verbose:
            print_features(cqt_normalized, actual_frames)
        
        return cqt_normalized
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def flatten_features(features: np.ndarray, order: str = 'C') -> np.ndarray:
    """
    Flatten 2D features to 1D array.
    
    The JavaScript version stores features as:
    [bin0_time0, bin0_time1, ..., bin1_time0, bin1_time1, ...]
    This is C-order (row-major) flattening.
    
    Args:
        features: 2D feature array [n_bins, time_frames]
        order: 'C' for row-major (JavaScript style), 'F' for column-major
        
    Returns:
        1D flattened array
    """
    return features.flatten(order=order)


def main():
    parser = argparse.ArgumentParser(
        description='Extract CQT features from audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_cqt_features.py audio.wav
  python extract_cqt_features.py audio.wav --no-normalize
  python extract_cqt_features.py audio.wav --save-npy output.npy
        """
    )
    
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('--sample-rate', type=int, default=SAMPLE_RATE,
                        help=f'Sample rate (default: {SAMPLE_RATE})')
    parser.add_argument('--hop-length', type=int, default=HOP_LENGTH,
                        help=f'Hop length in samples (default: {HOP_LENGTH})')
    parser.add_argument('--n-bins', type=int, default=N_BINS,
                        help=f'Number of frequency bins (default: {N_BINS})')
    parser.add_argument('--max-pad-len', type=int, default=MAX_CQT_PAD_LEN,
                        help=f'Max time frames (default: {MAX_CQT_PAD_LEN})')
    parser.add_argument('--fmin', type=str, default='C3',
                        help='Minimum frequency note (default: C3)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip normalization to 0-1 range')
    parser.add_argument('--save-npy', type=str, default=None,
                        help='Save features to .npy file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Validate audio path
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    
    # Convert note name to frequency if needed
    try:
        fmin = librosa.note_to_hz(args.fmin)
    except:
        fmin = float(args.fmin)
    
    print(f'\nCQT Feature Extraction Configuration:')
    print(f'  Sample Rate: {args.sample_rate} Hz')
    print(f'  Hop Length: {args.hop_length} samples')
    print(f'  Frequency Bins: {args.n_bins}')
    print(f'  Min Frequency: {fmin:.2f} Hz ({args.fmin})')
    print(f'  Max Time Frames: {args.max_pad_len}')
    print(f'  Normalize: {not args.no_normalize}')
    
    # Extract features
    features = extract_cqt_features(
        audio_path=str(audio_path),
        fmin=fmin,
        n_bins=args.n_bins,
        hop_length=args.hop_length,
        max_pad_len=args.max_pad_len,
        sample_rate=args.sample_rate,
        normalize=not args.no_normalize,
        verbose=not args.quiet
    )
    
    if features is None:
        return 1
    
    # Save to file if requested
    if args.save_npy:
        np.save(args.save_npy, features)
        print(f'Features saved to: {args.save_npy}')
    
    return 0


if __name__ == '__main__':
    exit(main())
