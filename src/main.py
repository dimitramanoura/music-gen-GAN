from data_processing import process_dataset
from train import train
from generate import generate_spectrograms, load_generator
from spectrogram_to_audio import convert_generated_spectrograms

# Define paths
AUDIO_DATASET_PATH = "path/to/your/audio/files"
SPECTROGRAM_OUTPUT_PATH = "spectrograms/"
GENERATED_OUTPUT_PATH = "generated_spectrograms/"
MODEL_PATH = "generator.pth"

if __name__ == "__main__":
    # Step 1: Process raw audio into spectrograms
    process_dataset([AUDIO_DATASET_PATH], SPECTROGRAM_OUTPUT_PATH)

    # Step 2: Train the GAN model
    train(SPECTROGRAM_OUTPUT_PATH)

    # Step 3: Generate spectrograms using the trained model
    generator = load_generator(MODEL_PATH)
    generate_spectrograms(generator, num_samples=5, output_dir=GENERATED_OUTPUT_PATH)

    # Step 4: Convert generated spectrograms back to audio
    convert_generated_spectrograms(GENERATED_OUTPUT_PATH)

    print("Pipeline complete! Check the generated spectrograms and audio files.")
