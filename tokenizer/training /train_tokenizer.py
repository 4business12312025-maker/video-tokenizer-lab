import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from encoder import Encoder
from decoder import Decoder


# Device selection
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# Models
encoder = Encoder().to(device)
decoder = Decoder().to(device)

# Optimizer
optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()),
    lr=1e-3
)

# Loss function
criterion = nn.MSELoss()

# Dataset transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Dataset
dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Data loader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):

    running_loss = 0.0

    for images, _ in loader:

        images = images.to(device)

        # Encode
        z = encoder(images)

        # Decode
        reconstructed = decoder(z)

        # Reconstruction loss
        loss = criterion(
            reconstructed,
            images
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Loss: {avg_loss:.6f}"
    )

print("Training complete.")loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for epoch in range(5):

    for images, _ in loader:

        images = images.to(device)

        z = encoder(images)

        reconstructed = decoder(z)

        loss = criterion(
            reconstructed,
            images
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"Epoch {epoch+1}, Loss: {loss.item():.4f}"
    )
