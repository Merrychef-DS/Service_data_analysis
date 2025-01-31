import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib

# Step 1: Load your dataset
data = pd.read_csv("Processed_FSR_SOlution_Keywords.csv")

# Step 2: Preprocess Service Descriptions (Text Processing with TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = tfidf_vectorizer.fit_transform(data['FSR_Solution'].fillna(''))

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

# Step 6: Train Part System Classifier
part_system_model = RandomForestClassifier(random_state=42)
part_system_model.fit(X_train_ps_resampled, y_train_ps_resampled)

# Save the model, encoder, and vectorizer
joblib.dump(part_system_model, 'part_system_model.pkl')
joblib.dump(part_system_encoder, 'part_system_encoder.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Print system-class mapping for reference
system_class_mapping = dict(zip(part_system_encoder.classes_, range(len(part_system_encoder.classes_))))
print("System to Class Mapping:")
for system, label in system_class_mapping.items():
    print(f"System: {system} -> Class: {label}")
