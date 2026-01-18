# Mini Transformer from Scratch (IMDB Sentiment Analysis)

##  Overview
This project implements a **Transformer encoder from scratch** using PyTorch for
sentiment classification on the IMDB movie reviews dataset.

##  Model Architecture
- Token Embedding
- Sinusoidal Positional Encoding
- Multi-Head Self Attention
- Feed Forward Network
- Residual Connections + LayerNorm
- Mean Pooling
- Classification Head

## Dataset
- IMDB Movie Reviews
- Binary sentiment classification (positive / negative)

## Training Details
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Batch Size: 32
- Epochs: 10

## How to Run
```bash
pip install torch pandas
python train.py
