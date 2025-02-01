import mlx.core as mx
import mlx.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        position = mx.arange(max_len)[:, None]
        div_term = mx.exp(mx.arange(0, dim, 2) * (-mx.log(10000.0) / dim))
        pe = mx.zeros((max_len, dim))
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        self.pe = pe[None, :, :]

    def __call__(self, x):
        return x + self.pe[:, :x.shape[1], :]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim=256, heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x
