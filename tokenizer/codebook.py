import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self,
                 num_embeddings=512,
                 embedding_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim
        )

    def forward(self, z):
        return z
