import torch
import torch.nn as nn

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_shape=(128, 128)):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape[0] * output_shape[1]),
            nn.Tanh()
        )
        self.output_shape = output_shape

    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, self.output_shape[0], self.output_shape[1])

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_shape=(128, 128)):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

if __name__ == "__main__":
    # Test model initialization
    generator = Generator()
    discriminator = Discriminator()
    print("Generator and Discriminator initialized successfully!")
