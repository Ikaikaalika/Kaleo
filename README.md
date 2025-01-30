# KALEO - Hawaiian Text-to-Speech (TTS) Project

## Overview
This project trains **Tacotron 2** and **FastSpeech 2** models for **Hawaiian speech synthesis**. It processes a dataset of **Hawaiian transcriptions and audio recordings** to generate high-quality text-to-speech (TTS) models. The project also includes a **HiFi-GAN vocoder** to convert **mel spectrograms to waveforms**.

## Features
- **Data Preprocessing**: Cleans Hawaiian text, converts to phonemes, and aligns text with speech.
- **Tacotron 2 Model**: Autoregressive text-to-mel model.
- **FastSpeech 2 Model**: Non-autoregressive TTS with duration modeling.
- **HiFi-GAN Vocoder**: Converts mel spectrograms to waveforms.
- **Inference Scripts**: Generate speech from text.
- **Deployment**: Serve TTS models via FastAPI.

## Project Structure
```
/kaleo
  ├── data/               # Dataset and text-audio alignments
  │   ├── wavs/           # Raw & processed audio files (16-bit WAV)
  │   ├── alignments/     # Duration alignments (for FastSpeech 2)
  │   ├── metadata.csv    # Transcriptions (Tacotron 2 format)
  │   ├── metadata_phoneme.csv  # Transcriptions + phonemes (FastSpeech 2)
  │   ├── lexicon.txt     # Custom phoneme dictionary (if needed)
  ├── preprocessing/      # Scripts for data processing
  │   ├── clean_text.py        # Normalize text, convert to phonemes
  │   ├── align_audio.py       # Generate phoneme-audio alignments (MFA)
  │   ├── preprocess_audio.py  # Convert, normalize, trim silence
  │   ├── extract_durations.py # Extract phoneme durations (FastSpeech 2)
  ├── training/           # Training scripts
  │   ├── train_tacotron2.py   # Train Tacotron 2
  │   ├── train_fastspeech2.py # Train FastSpeech 2
  │   ├── config_tacotron2.yaml  # Tacotron 2 config
  │   ├── config_fastspeech2.yaml # FastSpeech 2 config
  │   ├── checkpoints/      # Model checkpoints (saved during training)
  ├── inference/           # Run trained models
  │   ├── synthesize_tacotron2.py   # Generate speech (Tacotron 2)
  │   ├── synthesize_fastspeech2.py # Generate speech (FastSpeech 2)
  ├── vocoder/             # HiFi-GAN for mel-to-wave conversion
  │   ├── train_hifigan.py     # Train HiFi-GAN vocoder
  │   ├── inference_hifigan.py # Convert mel spectrograms to audio
  │   ├── checkpoints/      # HiFi-GAN checkpoints
  ├── notebooks/           # Jupyter notebooks for analysis
  │   ├── data_exploration.ipynb  # Visualize phoneme distributions
  │   ├── training_analysis.ipynb # Evaluate model training curves
  ├── models/              # Trained models for deployment
  │   ├── tacotron2_hawaiian.pth  # Trained Tacotron 2 model
  │   ├── fastspeech2_hawaiian.pth # Trained FastSpeech 2 model
  │   ├── hifigan_hawaiian.pth    # Trained HiFi-GAN vocoder
  ├── deployment/          # API and web demo
  │   ├── server.py            # FastAPI server
  │   ├── web_demo.py          # Simple web UI for testing TTS
  ├── requirements.txt      # Python dependencies
  ├── README.md             # Project documentation
```

## Setup Instructions
### 1️⃣ Install Dependencies
Ensure Python 3.8+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 2️⃣ Preprocess Dataset
Run preprocessing scripts in order:
```bash
python preprocessing/clean_text.py
python preprocessing/preprocess_audio.py
python preprocessing/align_audio.py
python preprocessing/extract_durations.py
```

### 3️⃣ Train Tacotron 2
```bash
python training/train_tacotron2.py
```

### 4️⃣ Train FastSpeech 2
```bash
python training/train_fastspeech2.py
```

### 5️⃣ Train HiFi-GAN Vocoder
```bash
python vocoder/train_hifigan.py
```

### 6️⃣ Generate Speech (Inference)
```bash
python inference/synthesize_fastspeech2.py --text "Aloha kāua e ke hoa"
```

### 7️⃣ Deploy as API
Run a FastAPI server:
```bash
uvicorn deployment.server:app --host 0.0.0.0 --port 8000
```
Test via:
```bash
curl "http://localhost:8000/tts/?text=Aloha kāua e ke hoa"
```

## Next Steps
- 🚀 **Fine-tune the models** with a larger Hawaiian dataset.
- 🎤 **Collect high-quality audio recordings** for better synthesis.
- 🌐 **Deploy the model as a web app** with Streamlit or Flask.

Need help? Open an issue or contribute! 🤙

