import librosa
import soundfile as sf
import numpy as np
import os

for path in os.listdir("C:/Users/asher/Desktop/Coding/sample-slicing/output_slices"):
    os.remove("C:/Users/asher/Desktop/Coding/sample-slicing/output_slices/" + path)

def apply_fade_out(audio, fade_out_samples):
    """Applies a linear fade-out to the audio signal.

    Args:
        audio: The audio signal to apply the fade-out to.
        fade_out_samples: The number of samples over which to apply the fade-out.
    
    Returns:
        The audio signal with the fade-out applied.
    """
    # Determine the actual number of samples to fade
    actual_fade_samples = min(len(audio), fade_out_samples)

    # Create a fade-out curve
    fade_out_curve = np.linspace(1, 0, actual_fade_samples)
    
    # Apply fade-out curve to the audio slice
    if actual_fade_samples > 0:
        audio[-actual_fade_samples:] *= fade_out_curve
    
    return audio

def normalize_audio(audio):
    """Normalizes the audio signal to have a maximum amplitude of 1.0.

    Args:
        audio: The audio signal to normalize.
    
    Returns:
        The normalized audio signal.
    """
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude > 0:
        audio = audio / max_amplitude
    return audio

def slice_audio(audio_file, output_dir, min_silence_len=0.1, overlap=0.0, sensitivity=1.0, fade_out_ms=50):
    """Slices audio into individual WAV files based on transients.

    Args:
        audio_file: Path to the input audio file.
        output_dir: Directory to save the output WAV files.
        min_silence_len: Minimum length of silence between transients (in seconds).
        overlap: Overlap between slices (in seconds).
        sensitivity: Sensitivity for transient detection (higher values increase sensitivity).
        fade_out_ms: Duration of the fade-out effect in milliseconds.
    """

    # Load audio file
    y, sr = librosa.load(audio_file)

    # Calculate the onset envelope
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)

    # Detect onset times with sensitivity control
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sr,
        backtrack=True,
        pre_max=int(20 * sensitivity),
        post_max=int(20 * sensitivity),
        pre_avg=int(100 * sensitivity),
        post_avg=int(100 * sensitivity),
        delta=0.2 * sensitivity,
    )
    onset_time = librosa.frames_to_time(onset_frames, sr=sr)

    # Calculate slice lengths based on onset times
    slice_lengths = np.diff(onset_time)

    # Filter out slices shorter than min_silence_len
    valid_indices = np.where(slice_lengths > min_silence_len)[0]
    onset_time = onset_time[valid_indices]
    slice_lengths = slice_lengths[valid_indices]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Slice and save audio
    for i, start_time in enumerate(onset_time):
        end_time = start_time + slice_lengths[i] - overlap
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        slice_audio = y[start_sample:end_sample]

        # Apply fade-out effect
        fade_samples = int(fade_out_ms / 1000 * sr)
        slice_audio = apply_fade_out(slice_audio, fade_samples)

        # Normalize the audio slice
        slice_audio = normalize_audio(slice_audio)

        output_file = os.path.join(output_dir, f"slice_{i}.wav")
        sf.write(output_file, slice_audio, sr)
# Example usage
audio_file = "audio.wav"
output_dir = "output_slices"
slice_audio(audio_file, output_dir, sensitivity=0.3, fade_out_ms=50)
