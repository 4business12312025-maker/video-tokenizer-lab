import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 32 → 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 16 → 8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256)
        )

    def forward(self, x):
        return self.model(x)
