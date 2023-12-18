# Chatbot on Google Colab

## Overview

This project is a simple chatbot implemented in Python using PyTorch and natural language processing techniques. The chatbot is trained to understand and respond to various user inputs based on predefined patterns and intents.

## Setup

### Run this project on Google Colab

1. **Upload Files to Google Colab:**
   - Upload the following files to your Google Colab environment:
     - `model.py`
     - `nltk_utils.py`
     - `intents.json`

2. **Create a new notebook:**
   - Run the following commands in a code cell to install the necessary libraries.
     ```python
     !pip install torch
     !pip install nltk
     !pip install sympy
     ```

3. **Copy the code:**
   - Copy the code from the file `train.py` and execute it. 
   - In another section of code, copy the code from `chat.py` and execute it.
   - Now you can interact with the chatbot named "Potato."

## Usage

### Training the Chatbot:

- If you want to train the chatbot with your dataset, modify the `intents.json` file with your patterns, responses, and tags.
- Run the training script or notebook cell to train the model.

### Interacting with the Chatbot:

- Input your messages in the designated code cell and execute it to receive responses from the chatbot.

### Customization:

- Feel free to customize the chatbot's behavior by modifying the `intents.json` file, adding more patterns, or adjusting the model architecture in `model.py`.

### Saving and Loading Models:

- If you train the model, save the trained model (`data.pth` or your specified filename).
- When using the chatbot in the future, load the trained model for better performance.
