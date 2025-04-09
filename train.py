import json

import numpy as np
import torch
import torch.nn as nn  # This is the missing import
from torch.utils.data import DataLoader, Dataset

from model import NeuralNet
from nltk_utils import bag_of_words, stem, tokenize

# Initialize NLTK
try:
    nltk.data.find("tokenizers/punkt")
except:
    import nltk

    nltk.download("punkt")

# Load and process data
with open("intents.json") as f:
    intents = json.load(f)

all_words, tags, xy = [], [], []
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))

all_words = [stem(w) for w in all_words if w not in ["?", "!", "."]]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X = np.array([bag_of_words(tokens, all_words) for tokens, _ in xy])
y = np.array([tags.index(tag) for _, tag in xy])

# Hyperparameters
EPOCHS = 20000
BATCH_SIZE = 8
LEARNING_RATE = 0.01
HIDDEN_SIZE = 16


class ChatDataset(Dataset):
    def __init__(self):
        self.x_data = torch.FloatTensor(X)
        self.y_data = torch.LongTensor(y)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# Initialize
dataset = ChatDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(len(X[0]), HIDDEN_SIZE, len(tags)).to(device)
criterion = nn.CrossEntropyLoss()  # This requires the nn import
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    for words, labels in loader:
        words, labels = words.to(device), labels.to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save model
torch.save(
    {
        "model_state": model.state_dict(),
        "input_size": len(X[0]),
        "hidden_size": HIDDEN_SIZE,
        "output_size": len(tags),
        "all_words": all_words,
        "tags": tags,
    },
    "data.pth",
)

print(f"Training complete. Model saved to data.pth")
