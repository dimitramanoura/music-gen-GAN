import numpy as np
import librosa
import librosa.display
import scipy.signal
import soundfile as sf
import os

# Convert spectrogram back to audio
def spectrogram_to_audio(spectrogram, sr=22050, n_fft=1024, hop_length=512, win_length=1024):
    audio = librosa.griffinlim(spectrogram, 
                                n_iter=32, 
                                hop_length=hop_length, 
                                win_length=win_length, 
                                window='hann')
    return audio


# Process all generated spectrograms
def convert_generated_spectrograms(input_dir="generated_spectrograms/", output_dir="generated_audio/"):
    os.makedirs(output_dir, exist_ok=True)
    spectrogram_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    for file in spectrogram_files:
        spectrogram = np.load(os.path.join(input_dir, file))
        audio = spectrogram_to_audio(spectrogram)
        scaled_audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
        sf.write(os.path.join(output_dir, file.replace(".npy", ".wav")), scaled_audio, 22050)

if __name__ == "__main__":
    convert_generated_spectrograms()
