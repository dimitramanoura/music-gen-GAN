import numpy as np
import matplotlib.pyplot as plt

# Normalize spectrogram values to range [-1, 1]
def normalize_spectrogram(spectrogram):
    return (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min()) * 2 - 1

# Denormalize back to original range
def denormalize_spectrogram(spectrogram, original_min, original_max):
    return (spectrogram + 1) / 2 * (original_max - original_min) + original_min

# Plot and save a spectrogram
def plot_spectrogram(spectrogram, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', cmap='magma', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Generated Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Example usage with random data
    sample_spectrogram = np.random.rand(128, 128)
    plot_spectrogram(sample_spectrogram)
