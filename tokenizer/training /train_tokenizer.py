import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from encoder import Encoder
from decoder import Decoder


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

encoder = Encoder().to(device)
decoder = Decoder().to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()),
    lr=1e-3
)

criterion = nn.MSELoss()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
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
