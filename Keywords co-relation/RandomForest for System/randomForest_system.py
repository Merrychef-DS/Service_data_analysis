import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter

# Step 1: Load your dataset
data = pd.read_csv("Processed_FSR_SOlution_Keywords.csv")

# Step 2: Preprocess Service Descriptions (Text Processing with TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = tfidf_vectorizer.fit_transform(data['FSR_Solution'].fillna(''))

unique_systems_before_encoding = data['System'].unique()
print("\nUnique Systems from CSV (After Normalization, Before Encoding):")
for idx, system in enumerate(unique_systems_before_encoding):
    print(f"Serial: {idx}, System: {system}")

# Step 3: Encode Part System (Target for Stage 1: Part System Prediction)
part_system_encoder = LabelEncoder()
y_part_system = part_system_encoder.fit_transform(data['System'])

# Step 3.1: Check class distribution
class_counts = Counter(y_part_system)
print("Original class distribution:", class_counts)

# Step 3.2: Handle rare classes by converting all labels to strings
threshold = 3  # Define a threshold for minimum samples
rare_classes = [cls for cls, count in class_counts.items() if count < threshold]

if rare_classes:
    print(f"Classes {rare_classes} have fewer than {threshold} samples. Combining into 'Other'.")

    # Convert class labels to strings and replace rare classes with 'Other'
    y_part_system = pd.Series(y_part_system).apply(lambda x: 'Other' if x in rare_classes else str(x))

# Step 3.3: Re-encode labels using LabelEncoder after conversion to strings
part_system_encoder = LabelEncoder()
y_part_system = part_system_encoder.fit_transform(y_part_system)

# Step 4: Split Data for Stage 1: Part System Prediction
X_train_ps, X_test_ps, y_train_ps, y_test_ps = train_test_split(
    X_text, y_part_system, test_size=0.2, random_state=42, stratify=y_part_system
)

# Step 5: Handle Imbalanced Data (Optional) - SMOTE to balance the Part System classes
min_samples = min(Counter(y_train_ps).values())
smote_neighbors = min_samples - 1 if min_samples > 1 else 1
print(f"Using SMOTE with n_neighbors={smote_neighbors}")

smote = SMOTE(random_state=42, k_neighbors=smote_neighbors)
X_train_ps_resampled, y_train_ps_resampled = smote.fit_resample(X_train_ps, y_train_ps)

# Verify new class distribution
resampled_class_counts = Counter(y_train_ps_resampled)
print("Resampled class distribution:", resampled_class_counts)

# Step 6: Train Part System Classifier (Stage 1: Predict Part System)
part_system_model = RandomForestClassifier(random_state=42)
part_system_model.fit(X_train_ps_resampled, y_train_ps_resampled)

# Step 7: Evaluate Part System Model
unique_labels_test = sorted(set(y_test_ps))

# Get the actual class names from the encoder for the test set
target_names = part_system_encoder.inverse_transform(unique_labels_test)

# Print the unique labels to debug
print(f"Unique labels in y_test_ps: {unique_labels_test}")
print(f"Target names: {target_names}")

# Manually specify the labels parameter to ensure all classes are accounted for
y_pred_ps = part_system_model.predict(X_test_ps)

# Get the full set of labels from the encoder (after merging rare classes)
labels = sorted(set(y_train_ps_resampled))  # Ensure these match the labels in the resampled training set
full_target_names = part_system_encoder.inverse_transform(labels)

# Now generate the classification report with the correct l
print("Part System Prediction Report:\n", classification_report(
    y_test_ps, y_pred_ps, target_names=full_target_names, labels=labels
))
system_class_mapping = dict(zip(part_system_encoder.classes_, range(len(part_system_encoder.classes_))))
print("System to Class Mapping:")
for system, label in system_class_mapping.items():
    print(f"System: {system} -> Class: {label}")
# Step 8: Function to Predict Part System
def predict_part_system(service_description):
    processed_text = tfidf_vectorizer.transform([service_description])
    predicted_system = part_system_model.predict(processed_text)
    return part_system_encoder.inverse_transform(predicted_system)

# Example usage
example_description = "Replace the faulty sensor in the cooling system."
predicted_system = predict_part_system(example_description)
print(f"Predicted System: {predicted_system[0]}")


