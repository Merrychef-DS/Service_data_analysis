from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import json
import sys

# Load the system label mapping
with open('system_label_mapping.json', 'r') as f:
    system_label_mapping = json.load(f)

# Load the tokenizer and model from the directory where they were saved
tokenizer = DistilBertTokenizer.from_pretrained('distilbert_system_classifier')
model = DistilBertForSequenceClassification.from_pretrained('distilbert_system_classifier')


def classify_text(input_text):
    # Tokenize the text
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Map the predicted label to the system
    predicted_label = list(system_label_mapping.keys())[predictions.item()]

    # Display a fancy result output
    print("\n-------------------------------------")
    print(f"Input Text: {input_text}")
    print(f"Predicted System: {predicted_label}")
    print("-------------------------------------\n")


if __name__ == "__main__":
    while True:
        # Ask the user for input text
        user_input = input("\nEnter the text to classify (or type 'exit' to quit): ")

        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Exiting the program. Have a great day!")
            sys.exit()

        # Classify the user input text
        classify_text(user_input)
