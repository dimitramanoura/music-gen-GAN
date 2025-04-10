import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Generator, Discriminator, weights_init, nz, ngpu, nc, ngf, ndf

def train(data_path):
    # Training parameters
    lr = 0.0002  # Learning rate
    num_epochs = 150
    batch_size = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Generator and Discriminator
    generator = Generator(ngpu).to(device)
    discriminator = Discriminator(ngpu).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss function & optimizers
    criterion = nn.BCELoss()
    optimizerG = torch.optim.SGD(generator.parameters(), lr=lr)
    optimizerD = torch.optim.SGD(discriminator.parameters(), lr=lr)

    #Load dataset
    spectrogram_files = sorted(glob.glob(os.path.join(data_path, '*.npy')))
    spectrograms = [torch.tensor(np.load(f), dtype=torch.float32).unsqueeze(0) for f in spectrogram_files]
    spectrograms_tensor = torch.stack(spectrograms)
    dataset = torch.utils.data.TensorDataset(spectrograms_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Fixed noise for visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Training Loop
    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Train Discriminator
            ############################
            discriminator.zero_grad()
            
            # Real batch
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
            output_real = discriminator(real_images).view(-1)
            loss_real = criterion(output_real, real_labels)
            loss_real.backward()

            # Fake batch
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = generator(noise)
            fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)
            output_fake = discriminator(fake_images.detach()).view(-1)
            loss_fake = criterion(output_fake, fake_labels)
            loss_fake.backward()
            
            lossD = loss_real + loss_fake
            optimizerD.step()

            ############################
            # (2) Train Generator
            ############################
            generator.zero_grad()
            output_fake = discriminator(fake_images).view(-1)
            lossG = criterion(output_fake, real_labels)
            lossG.backward()
            optimizerG.step()

            # Print progress
            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {lossD:.4f} Loss_G: {lossG:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                grid = vutils.make_grid(fake, padding=2, normalize=True)
                os.makedirs("generated_images", exist_ok=True)
                vutils.save_image(grid, f"generated_images/epoch_{epoch + 1}.png")

    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Training Finished!")
