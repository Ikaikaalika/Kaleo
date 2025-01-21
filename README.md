# Kaleo
A Speech-to-Text and Text-to-Speech Project for ʻŌlelo Hawaiʻi. Models will be used in other projects like translators and LLMʻs.

Ka Leo means that sound in ʻŌlelo Hawaiʻi. This is a passion project thats uses multiple public data sources to build the STT and TTS Models. Because ʻŌlelo Hawaiʻi is an endangered language the TTS model will is prioritized as this will help learners hear the language and keep it alive.

Datasets used:

Kaniʻaina


# TTS for Hawaiian Using Transformers

This repository contains code for developing Text-to-Speech models specifically for the Hawaiian language using Transformer architectures. The project includes data cleaning from mixed language audio files and training custom TTS models.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup](#setup)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to:
- Clean audio data containing both Hawaiian and English.
- Train transformer-based TTS models for Hawaiian speech synthesis.
- Provide a framework for expanding to other low-resource languages.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tts-hawaiian-transformer.git
   cd tts-hawaiian-transformer

2. Install dependencies:
pip install -r requirements.txt
Usage
Data Processing

Place your raw audio files in data/raw/.
Run data cleaning:
bash

python src/data_processing/clean_audio.py
Training Models

For training the Transformer TTS model:
bash

python scripts/train_transformer.py
Data
Raw Data: Audio files with both Hawaiian and English speech.
Cleaned Data: Audio segments isolated by language.
Models
Transformer TTS: Located in src/models/transformer_tts.py.
Contributing
Contributions are welcome! Please read the contribution guidelines before starting.

License
[Choose an appropriate license]