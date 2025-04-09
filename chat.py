import json
import random

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, sanitize_input, tokenize

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.load("data.pth", map_location=device)
model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(
    device
)
model.load_state_dict(data["model_state"])
model.eval()

# Load intents
with open("intents.json") as f:
    intents = json.load(f)

bot_name = "CommerceBot"
print("Dynamics 365 Commerce Assistant (type 'quit' to exit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Validate input
    clean_input = sanitize_input(user_input)
    if not clean_input:
        print(f"{bot_name}: Please ask a Commerce-related question.")
        continue

    # Process input
    tokens = tokenize(clean_input)
    X = bag_of_words(tokens, data["all_words"])
    X = torch.FloatTensor(X).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(X)
    prob = torch.softmax(output, dim=1)[0]
    conf, pred = torch.max(prob, 0)

    # Respond
    if conf.item() > 0.85:
        tag = data["tags"][pred.item()]
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(
            f"{bot_name}: I didn't understand. Try asking about:\n- Commerce setup\n- POS configuration\n- Payment integration"
        )
