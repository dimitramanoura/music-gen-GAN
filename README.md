# 🎶 Deep Music GAN

This project uses a **DCGAN** (Deep Convolutional GAN) to generate spectrograms from audio data and reconstruct them back into sound.

## 📌 Project Structure
```
deep-music-gan/
├── src/
│   ├── data_processing.py        # Audio to spectrogram conversion, normalization
│   ├── model.py                  # GAN model (generator & discriminator)
│   ├── train.py                  # Training script for the GAN
│   ├── generate.py               # Generates spectrograms using trained GAN
│   ├── spectrogram_to_audio.py    # Converts spectrograms back to audio
│   ├── utils.py                   # Helper functions (e.g., normalization, visualization)
│   ├── main.py                    # Main script to run the entire process
├── notebooks/
│   └── thesis_notebook.ipynb      # Original Colab notebook
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                       # Ignore unnecessary files
```

## 🚀 Installation

1. **Clone the repository** (or download the ZIP manually).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔥 Usage

### **1️⃣ Convert Audio to Spectrograms**
```bash
python src/data_processing.py
```
### **2️⃣ Train the GAN**
```bash
python src/train.py
```
### **3️⃣ Generate Spectrograms**
```bash
python src/generate.py
```
### **4️⃣ Convert Spectrograms Back to Audio**
```bash
python src/spectrogram_to_audio.py
```

## 📊 Results
The trained GAN generates spectrograms, which can be reconstructed into music-like audio.
