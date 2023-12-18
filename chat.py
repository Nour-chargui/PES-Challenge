import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from sympy import sympify, parse_expr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Potato"
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Check if the model is confident and the tag is not a fallback
    if prob.item() > 0.75 and tag != "fallback":
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}")

                # Additional logic for specific game-related responses
                if tag == "word_association" and "Here's mine." in response:
                    user_response = input("You: ")
                    print(f"{bot_name}: Your word: {user_response}")
                elif tag == "math_game" and "What is" in response:
                    math_game_parts = response.split(":")
                    if len(math_game_parts) > 1:
                        math_game_question = math_game_parts[1].strip()
                        correct_answer = parse_expr(math_game_question)
                        print(f"{bot_name}: {math_game_question}")
                break  # Break after finding the matching intent
    else:
        # If the tag is a fallback, respond with a generic message
        print(f"{bot_name}: I'm not sure about that. Let's talk about something else!")
