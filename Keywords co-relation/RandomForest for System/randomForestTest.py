import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load the saved model, encoder, and vectorizer
part_system_model = joblib.load('part_system_model.pkl')
part_system_encoder = joblib.load('part_system_encoder.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load your test dataset
data = pd.read_csv("Processed_FSR_SOlution_Keywords.csv")

# Preprocess Service Descriptions (Text Processing with TF-IDF)
X_text = tfidf_vectorizer.transform(data['FSR_Solution'].fillna(''))
print(part_system_encoder)
print(data['System'])
# Encode Part System for the test data
y_part_system = part_system_encoder.transform(data['System'])
print(y_part_system)

# Split Data for Stage 1: Part System Prediction
X_train_ps, X_test_ps, y_train_ps, y_test_ps = train_test_split(
    X_text, y_part_system, test_size=0.2, random_state=42, stratify=y_part_system
)

# Predict using the loaded model
y_pred_ps = part_system_model.predict(X_test_ps)

# Get the full set of labels from the encoder for evaluation
unique_labels_test = sorted(set(y_test_ps))
target_names = part_system_encoder.inverse_transform(unique_labels_test)

# Generate the classification report
print("Part System Prediction Report:\n", classification_report(
    y_test_ps, y_pred_ps, target_names=target_names
))

# Function to Predict Part System from new input
def predict_part_system(service_description):
    processed_text = tfidf_vectorizer.transform([service_description])
    predicted_system = part_system_model.predict(processed_text)
    return part_system_encoder.inverse_transform(predicted_system)

# Example usage
example_description = "Replace the faulty sensor in the cooling system."
predicted_system = predict_part_system(example_description)
print(f"Predicted System: {predicted_system[0]}")
