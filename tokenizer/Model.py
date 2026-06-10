
from encoder import Encoder
from decoder import Decoder

class Autoencoder:
    def __init__(self, device):
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
