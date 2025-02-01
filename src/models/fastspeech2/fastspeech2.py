from models.fastspeech2.encoder import PositionalEncoding, TransformerEncoderLayer
from models.fastspeech2.variance_predictor import VariancePredictor
from models.fastspeech2.length_regulator import length_regulator
import mlx.nn as nn
import mlx.core as mx

class FastSpeech2(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, mel_dim=80, num_layers=6, heads=4):
        super().__init__()
        self.encoder = nn.Sequential(
            PositionalEncoding(hidden_dim),
            *[TransformerEncoderLayer(hidden_dim, heads) for _ in range(num_layers)]
        )
        self.duration_predictor = VariancePredictor(hidden_dim)
        self.pitch_predictor = VariancePredictor(hidden_dim)
        self.energy_predictor = VariancePredictor(hidden_dim)
        self.decoder = nn.Sequential(
            PositionalEncoding(hidden_dim),
            *[TransformerEncoderLayer(hidden_dim, heads) for _ in range(num_layers)]
        )
        self.mel_proj = nn.Linear(hidden_dim, mel_dim)

    def __call__(self, x):
        encoder_out = self.encoder(x)
        durations = mx.relu(self.duration_predictor(encoder_out))
        pitch = self.pitch_predictor(encoder_out)
        energy = self.energy_predictor(encoder_out)

        regulated = length_regulator(encoder_out, durations.squeeze(-1).astype(int))
        decoder_out = self.decoder(regulated)
        mel_out = self.mel_proj(decoder_out)
        return mel_out
