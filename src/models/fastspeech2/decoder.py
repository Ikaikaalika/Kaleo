import mlx.core as mx
import mlx.nn as nn
from models.fastspeech2.encoder import PositionalEncoding, TransformerEncoderLayer

class TransformerDecoder(nn.Module):
    def __init__(self, dim=256, num_layers=6, heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.pos_encoding = PositionalEncoding(dim)
        self.layers = nn.Sequential(
            *[TransformerEncoderLayer(dim, heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x):
        x = self.pos_encoding(x)
        x = self.layers(x)
        x = self.norm(x)
        return x
