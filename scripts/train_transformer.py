from src.models.transformer_tts import TransformerTTS
from mlx.optimizers import Adam
import mlx.data as mx_data

class TTS_Dataset(mx_data.Dataset):
    def __init__(self, texts, audios):
        self.texts = texts
        self.audios = audios

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.audios[idx]

# Assuming you have prepared your data
dataset = TTS_Dataset(texts, audios)
dataloader = mx_data.DataLoader(dataset, batch_size=32, shuffle=True)

model = TransformerTTS(vocab_size=1000)
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for text, audio in dataloader:
        tgt_mask = mx.triu(mx.ones((text.size(1), text.size(1))), diagonal=1).bool()
        predictions = model(text, text, tgt_mask=tgt_mask)
        loss = mx.mean((predictions - audio) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()