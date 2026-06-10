from encoder import Encoder
from decoder import Decoder
from codebook import Codebook

class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.codebook = Codebook()
        self.decoder = Decoder()

    def forward(self, x):

        z = self.encoder(x)

        z_q = self.codebook(z)

        x_hat = self.decoder(z_q)

        return x_hat
