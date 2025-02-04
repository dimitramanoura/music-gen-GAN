import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import Generator

# Hyperparameters
LATENT_DIM = 100
IMG_SHAPE = (128, 128)

# Load trained generator
def load_generator(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim=LATENT_DIM, output_shape=IMG_SHAPE).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    return generator

# Generate and save spectrograms
def generate_spectrograms(generator, num_samples=10, output_dir="generated_spectrograms/"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        z = torch.randn(num_samples, LATENT_DIM, device=device)
        gen_imgs = generator(z).cpu().numpy().squeeze()

        for i, img in enumerate(gen_imgs):
            np.save(os.path.join(output_dir, f"generated_{i}.npy"), img)
            plt.imsave(os.path.join(output_dir, f"generated_{i}.png"), img, cmap="magma")

if __name__ == "__main__":
    generator = load_generator("generator.pth")
    generate_spectrograms(generator)
