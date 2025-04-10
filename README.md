# Dynamics 365 Commerce Assistant Chatbot

## Project Overview
A neural network-powered chatbot designed to answer Dynamics 365 Commerce questions, using PyTorch for machine learning and NLTK for natural language processing.

## Key Features
- Intent classification with 85% confidence threshold
- Special handling for commerce terminology
- Custom tokenization and stemming
- Simple JSON-based training data format

## Installation
```bash
pip install torch nltk numpy
python -c "import nltk; nltk.download('punkt')"

## File Structure

.
├── train.ipynb       # Model training script
├── chat.py           # Interactive chat interface
├── model.py          # Neural network architecture
├── nltk_utils.py     # NLP processing utilities
├── intents.json      # Training data (patterns & responses)
├── data.pth          # Trained model weights
└── README.md

