import mlx.core as mx
from mlx.nn import Module, Linear, LayerNorm, Embedding
from mlx.nn.layers import MultiHeadAttention

class TransformerBlock(Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output = self.attn(x, x, x, mask=mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff2(mx.relu(self.ff(x)))
        x = self.norm2(x + ff_output)
        return x

class TransformerTTS(Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.encoder_layers = [TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(num_encoder_layers)]
        self.decoder_layers = [TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(num_decoder_layers)]
        self.output_layer = Linear(d_model, 80)  # Assuming 80 Mel bins

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_mask)

        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(tgt, tgt_mask)

        return self.output_layer(tgt)