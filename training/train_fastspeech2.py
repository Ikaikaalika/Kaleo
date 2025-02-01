from models.fastspeech2.fastspeech2 import FastSpeech2
from training.utils import calculate_loss
import mlx.core as mx
import mlx.optimizers as optim

# Initialize model and optimizer
model = FastSpeech2()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training data
x_train = mx.random.uniform((4, 100, 512))  # Phoneme embeddings
y_train = mx.random.uniform((4, 300, 80))   # Mel-spectrogram

# Training loop
for epoch in range(100):
    def loss_fn():
        y_pred = model(x_train)
        return calculate_loss(y_pred, y_train)

    loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
    optimizer.update(model.parameters(), grads)
    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
