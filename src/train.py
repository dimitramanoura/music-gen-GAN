import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from model import Generator, Discriminator

# Hyperparameters
LATENT_DIM = 100
IMG_SHAPE = (128, 128)
EPOCHS = 10000
BATCH_SIZE = 32
LEARNING_RATE = 0.0002

# Load Data
def load_data(dataset_path):
    files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.npy')]
    data = [np.load(f) for f in files]
    data = np.array(data)
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    return data

# Training Loop
def train(dataset_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim=LATENT_DIM, output_shape=IMG_SHAPE).to(device)
    discriminator = Discriminator(input_shape=IMG_SHAPE).to(device)

    adversarial_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Load Data
    dataset = load_data(dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(device)

            # Adversarial ground truths
            valid = torch.ones((real_imgs.size(0), 1), device=device, dtype=torch.float32)
            fake = torch.zeros((real_imgs.size(0), 1), device=device, dtype=torch.float32)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(real_imgs.size(0), LATENT_DIM, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] D loss: {d_loss.item()} | G loss: {g_loss.item()}")

    # Save models
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")

if __name__ == "__main__":
    train("spectrograms/")
