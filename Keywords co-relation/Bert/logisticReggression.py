import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import json

# Load the data
data_list = []
with open("Grouped_Part_and_Keywords_for_n=3.json", "r") as file:
    for line in file:
        data_list.append(json.loads(line.strip()))

# Convert the list of dictionaries to a DataFrame
data = pd.DataFrame(data_list)
print(data.columns)

# Prepare features and labels
X = data['Keywords for n=2'].apply(lambda x: ' '.join(x))  # Join keywords into a single string
y = data['Part Info']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Vectorize the keywords using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the TF-IDF vectorizer, label encoder, and classifier
joblib.dump(vectorizer, './tfidf_vectorizer.pkl')
joblib.dump(label_encoder, './label_encoder.pkl')
joblib.dump(classifier, './classifier_model.pkl')
