import mlx.nn as nn
import mlx.core as mx

class VariancePredictor(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Conv1d(dim, 1, kernel_size=3, padding=1)
        )

    def __call__(self, x):
        x = mx.swapaxes(x, 1, 2)
        out = self.conv(x)
        return mx.swapaxes(out, 1, 2)
