import numpy as np
import librosa
import librosa.display
import scipy.signal
import os

# Convert spectrogram back to audio
def spectrogram_to_audio(spectrogram, sr=22050, hop_length=512, win_length=1024):
    _, audio = scipy.signal.istft(spectrogram, fs=sr, nperseg=win_length, noverlap=hop_length)
    return audio

# Process all generated spectrograms
def convert_generated_spectrograms(input_dir="generated_spectrograms/", output_dir="generated_audio/"):
    os.makedirs(output_dir, exist_ok=True)
    spectrogram_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    for file in spectrogram_files:
        spectrogram = np.load(os.path.join(input_dir, file))
        audio = spectrogram_to_audio(spectrogram)
        librosa.output.write_wav(os.path.join(output_dir, file.replace(".npy", ".wav")), audio, sr=22050)

if __name__ == "__main__":
    convert_generated_spectrograms()
