
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
import sys

# Load the pre-trained model, tokenizer, and label mapping
model_save_path = 'distilbert_system_classifier_balanced'
model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_save_path)
system_label_mapping = joblib.load('system_label_mapping_balanced.pkl')

# Reverse the system_label_mapping to map labels back to system names
label_to_system_mapping = {v: k for k, v in system_label_mapping.items()}

# Function to predict the system with probabilities
def predict_top_systems(service_description, top_n=3):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Tokenize the input text
        inputs = tokenizer(service_description, return_tensors='pt', padding="max_length", truncation=True, max_length=64)

        # Get model predictions
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()

        # Sort probabilities and get top N predictions
        top_predictions = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)[:top_n]

        # Map the predicted class labels back to the system names
        top_results = [(label_to_system_mapping.get(idx, "Unknown"), prob) for idx, prob in top_predictions]
        return top_results

# Interactive terminal input for service description prediction with probabilities
def interactive_prediction():
    while True:
        user_input = input("\nEnter the service description (or type 'exit' to quit): ")

        if user_input.lower() == 'exit':
            print("Exiting the program. Have a great day!")
            sys.exit()

        top_predictions = predict_top_systems(user_input)
        print("\n-------------------------------------")
        print(f"Input Text: {user_input}")
        print("Top Predictions:")
        for system, prob in top_predictions:
            print(f"  {system}: {prob:.2%}")
        print("-------------------------------------\n")


def predict_systems_for_column_with_probs(file_path, column_name, output_file, top_n=3):

    data = pd.read_csv(file_path)
    valid_data = data[data[column_name].notna() & data[column_name].str.strip().astype(bool)]

    valid_data['Top Predictions'] = valid_data[column_name].apply(
        lambda desc: predict_top_systems(desc, top_n=top_n)
    )

    data.loc[valid_data.index, 'Top Predictions'] = valid_data['Top Predictions']
    data.to_csv(output_file, index=False)
    print(f"Prediction results saved to: {output_file}")

def interactive_csv_prediction():
    file_path = input("\nEnter the CSV file path: ")
    column_name = input("Enter the column name for service descriptions: ")
    output_file = input("Enter the output CSV file name (with .csv extension): ")

    predict_systems_for_column_with_probs(file_path, column_name, output_file)

if __name__ == "__main__":
    print("\nChoose an option:")
    print("1. Predict system for a service description")
    print("2. Predict systems for a CSV file")

    option = input("Enter your choice (1 or 2): ")

    if option == '1':
        interactive_prediction()
    elif option == '2':
        interactive_csv_prediction()
    else:
        print("Invalid option. Please enter 1 or 2.")

#
#
#
# import pandas as pd
#
# # Load the data
# file_path = 'matched_parts_summary_12-11.csv'
# data = pd.read_csv(file_path)
#
# # Columns to check for duplicates, including the "System" column
# columns_to_check = ['service_incident_summary', 'Keywords for n=1', 'Keywords for n=2', 'Keywords for n=3', 'System']
#
# # Drop duplicates based on the selected columns, keeping the first occurrence
# data_deduped = data.drop_duplicates(subset=columns_to_check, keep='first')
#
# # Save the cleaned data to a new file
# output_path = 'matched_parts_summary_15-11.csv'
# data_deduped.to_csv(output_path, index=False)
#
# print("Duplicates removed successfully, considering the 'System' column. Cleaned file saved as:", output_path)
