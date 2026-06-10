import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from encoder import Encoder
from decoder import Decoder

# ------------------
# Device
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ------------------
# Models
# ------------------
encoder = Encoder().to(device)
decoder = Decoder().to(device)

params = list(encoder.parameters()) + list(decoder.parameters())

optimizer = torch.optim.Adam(params, lr=1e-3)
criterion = nn.MSELoss()

# ------------------
# Dataset
# ------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(train_data, batch_size=32, shuffle=True)

# ------------------
# Training
# ------------------
epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for images, _ in loader:
        images = images.to(device)

        # forward
        z = encoder(images)
        recon = decoder(z)

        loss = criterion(recon, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.6f}")

print("Training complete.")
