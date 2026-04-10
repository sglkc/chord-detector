#!/usr/bin/env python3
"""
Generate validation audio files by combining individual chord samples.
Outputs WAV audio files with MIREX format annotation text files.

MIREX annotation format:
start end chord
0.000 1.234 C:maj

Requires: pydub (uv pip install pydub) and ffmpeg
"""

import random
from pathlib import Path
from pydub import AudioSegment

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================

# Number of chords to combine per output audio file
CHORDS_PER_AUDIO = 5

# Number of output audio files to generate
AUDIO_COUNT = 5

# Output file prefix (files will be named: {PREFIX}-1.wav, {PREFIX}-2.wav, etc.)
OUTPUT_PREFIX = "validation"

# Types to include: "normal", "delay", or both
SELECTED_TYPES = ["normal", "delay"]

# Selected chords to include (leave empty [] to include all available)
# Format: ["C_major_4", "A_minor_4", "G#_diminished_4", ...]
SELECTED_CHORDS = []

# Gap between chords in seconds (silence between chord samples)
GAP_SECONDS = 0

# Random seed for reproducibility (set to None for random each run)
RANDOM_SEED = None

# Normalize audio to balance volume levels (recommended when mixing different types)
# Set to True to enable normalization
NORMALIZE_AUDIO = True

# Target dBFS for normalization (0 = max volume, -3 to -6 is typical for headroom)
# Lower values = quieter, higher values (closer to 0) = louder
TARGET_DBFS = -6.0

# =============================================================================
# CHORD MAPPING FOR MIREX FORMAT
# =============================================================================

# Mapping from folder chord type to MIREX chord type
CHORD_TYPE_MAP = {
    "major": "maj",
    "minor": "min",
    "diminished": "dim",
}

# Mapping from folder note name to MIREX note name
NOTE_MAP = {
    "C": "C",
    "C#": "C#",
    "D": "D",
    "D#": "D#",
    "E": "E",
    "F": "F",
    "F#": "F#",
    "G": "G",
    "G#": "G#",
    "A": "A",
    "A#": "A#",
    "B": "B",
}

# =============================================================================
# SCRIPT LOGIC
# =============================================================================

def parse_chord_folder_name(folder_name: str) -> tuple[str, str, str]:
    """
    Parse chord folder name into components.
    Example: "C#_minor_4" -> ("C#", "minor", "4")
    """
    parts = folder_name.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"Invalid chord folder name: {folder_name}")


def to_mirex_chord(folder_name: str) -> str:
    """
    Convert folder chord name to MIREX format.
    Example: "C#_minor_4" -> "C#:min"
    """
    root, chord_type, _ = parse_chord_folder_name(folder_name)
    mirex_root = NOTE_MAP.get(root, root)
    mirex_type = CHORD_TYPE_MAP.get(chord_type, chord_type)
    return f"{mirex_root}:{mirex_type}"


def get_available_chords(base_dir: Path, types: list[str]) -> dict[str, list[Path]]:
    """
    Get all available chord samples organized by chord name.
    Returns: {chord_folder_name: [list of wav file paths]}
    """
    chords = {}
    
    for audio_type in types:
        type_dir = base_dir / audio_type
        if not type_dir.exists():
            print(f"Warning: Directory {type_dir} does not exist, skipping...")
            continue
            
        for chord_dir in type_dir.iterdir():
            if not chord_dir.is_dir():
                continue
            
            chord_name = chord_dir.name
            if chord_name not in chords:
                chords[chord_name] = []
            
            # Get all wav files in this chord directory
            wav_files = list(chord_dir.glob("*.wav"))
            chords[chord_name].extend(wav_files)
    
    return chords


def normalize_audio(audio: AudioSegment, target_dbfs: float = -3.0) -> AudioSegment:
    """
    Normalize audio to a target dBFS level.
    This ensures consistent volume across all audio segments.
    """
    change_in_dbfs = target_dbfs - audio.dBFS
    return audio.apply_gain(change_in_dbfs)


def load_audio(file_path: Path, normalize: bool = False, target_dbfs: float = -3.0) -> AudioSegment:
    """
    Load audio file using pydub (auto-detects format via ffmpeg).
    Works with WebM, WAV, MP3, and other formats.
    
    Args:
        file_path: Path to the audio file
        normalize: If True, normalize the audio to target_dbfs
        target_dbfs: Target volume level in dBFS (only used if normalize=True)
    """
    audio = AudioSegment.from_file(str(file_path))
    if normalize:
        audio = normalize_audio(audio, target_dbfs)
    return audio


def combine_audio_files(audio_files: list[Path], gap_ms: int) -> tuple[AudioSegment, list[tuple]]:
    """
    Combine multiple audio files with gaps between them.
    Returns: (combined_audio, [(start, end, chord_name), ...])
    """
    combined = AudioSegment.empty()
    annotations = []
    current_time_ms = 0
    
    silence = AudioSegment.silent(duration=gap_ms)
    
    for i, audio_path in enumerate(audio_files):
        # Load audio segment (with optional normalization)
        audio = load_audio(audio_path, normalize=NORMALIZE_AUDIO, target_dbfs=TARGET_DBFS)
        duration_ms = len(audio)
        
        # Get chord name from file path
        chord_folder = audio_path.parent.name
        mirex_chord = to_mirex_chord(chord_folder)
        
        # Add annotation (convert ms to seconds)
        start_time = current_time_ms / 1000.0
        end_time = (current_time_ms + duration_ms) / 1000.0
        annotations.append((start_time, end_time, mirex_chord))
        
        # Add audio
        combined += audio
        current_time_ms += duration_ms
        
        # Add gap (silence) if not the last file
        if i < len(audio_files) - 1 and gap_ms > 0:
            combined += silence
            current_time_ms += gap_ms
    
    return combined, annotations


def write_mirex_annotation(output_path: Path, annotations: list[tuple]):
    """Write MIREX format annotation file."""
    with open(output_path, 'w') as f:
        for start, end, chord in annotations:
            f.write(f"{start:.3f} {end:.3f} {chord}\n")


def main():
    # Set random seed if specified
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    
    # Setup paths
    script_dir = Path(__file__).parent
    datasets_dir = script_dir / "datasets"
    output_dir = datasets_dir / "validation"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get available chords
    print("Scanning available chord samples...")
    available_chords = get_available_chords(datasets_dir, SELECTED_TYPES)
    
    if not available_chords:
        print("Error: No chord samples found!")
        return
    
    # Filter chords if SELECTED_CHORDS is specified
    if SELECTED_CHORDS:
        available_chords = {k: v for k, v in available_chords.items() if k in SELECTED_CHORDS}
    
    if not available_chords:
        print("Error: No matching chords found with the specified selection!")
        return
    
    print(f"Found {len(available_chords)} chord types with samples from types: {SELECTED_TYPES}")
    chord_names = list(available_chords.keys())
    
    # Generate audio files
    for i in range(1, AUDIO_COUNT + 1):
        print(f"\nGenerating audio {i}/{AUDIO_COUNT}...")
        
        # Select random chords
        selected_chord_names = random.choices(chord_names, k=CHORDS_PER_AUDIO)
        
        # Select random sample for each chord
        audio_files = []
        for chord_name in selected_chord_names:
            samples = available_chords[chord_name]
            audio_files.append(random.choice(samples))
        
        # Combine audio files (convert gap to milliseconds)
        try:
            gap_ms = int(GAP_SECONDS * 1000)
            combined_audio, annotations = combine_audio_files(audio_files, gap_ms)
        except Exception as e:
            print(f"Error combining audio files: {e}")
            continue
        
        # Write output files
        output_wav = output_dir / f"{OUTPUT_PREFIX}-{i}.wav"
        output_txt = output_dir / f"{OUTPUT_PREFIX}-{i}.txt"
        
        # Export as WAV format
        combined_audio.export(str(output_wav), format="wav")
        write_mirex_annotation(output_txt, annotations)
        
        print(f"  Created: {output_wav.name}")
        print(f"  Created: {output_txt.name}")
        print(f"  Chords: {[to_mirex_chord(af.parent.name) for af in audio_files]}")
        print(f"  Files:  {[af.name for af in audio_files]}")
    
    print(f"\nDone! Generated {AUDIO_COUNT} validation audio files in {output_dir}")


if __name__ == "__main__":
    main()
