import numpy as np
import librosa
import librosa.display
import scipy.signal
import os

# Function to load audio and split into segments
def load_and_segment_audio(file_path, segment_length=3, sample_rate=22050):
    y, sr = librosa.load(file_path, sr=sample_rate)
    segment_samples = segment_length * sr
    segments = [y[i : i + segment_samples] for i in range(0, len(y), segment_samples) if len(y[i : i + segment_samples]) == segment_samples]
    return segments, sr

# Function to compute spectrogram
def compute_spectrogram(audio_segment, sr):
    frequencies, times, spectrogram = scipy.signal.spectrogram(audio_segment, fs=sr, nperseg=1024, noverlap=512)
    return frequencies, times, spectrogram

# Function to process dataset and save spectrograms
def process_dataset(audio_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_path in audio_files:
        segments, sr = load_and_segment_audio(file_path)
        for i, segment in enumerate(segments):
            _, _, spectrogram = compute_spectrogram(segment, sr)
            np.save(os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_{i}.npy"), spectrogram)

if __name__ == "__main__":
    # Example usage (update with real paths)
    audio_files = ["path/to/audio1.wav", "path/to/audio2.wav"]
    process_dataset(audio_files, "spectrograms/")
