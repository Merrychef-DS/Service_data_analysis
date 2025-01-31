import pandas as pd
from sklearn.utils import resample
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import joblib
import numpy as np

# Load and clean the dataset
data = pd.read_csv("matched_parts_summary_15-11.csv")
cleaned_data = data.dropna(subset=['Keywords for n=3']).copy()
print("Dataset loaded and cleaned. Initial shape:", cleaned_data.shape)

# Balance the classes
class_counts = cleaned_data['System'].value_counts()
print("Initial class distribution:\n", class_counts)

min_class_samples = 500
majority_class = cleaned_data[cleaned_data['System'].isin(class_counts[class_counts >= min_class_samples].index)]
minority_class = cleaned_data[cleaned_data['System'].isin(class_counts[class_counts < min_class_samples].index)]
upsampled_minority = minority_class.groupby('System', group_keys=False).apply(
    lambda x: resample(x, replace=True, n_samples=min_class_samples, random_state=42)
)
balanced_data = pd.concat([majority_class, upsampled_minority], ignore_index=True)
print("Balanced class distribution:\n", balanced_data['System'].value_counts())

# Label mapping
# Standardize System column to lowercase to merge similar categories
balanced_data['System'] = balanced_data['System'].str.lower().str.strip()
print("Normalized System values:\n", balanced_data['System'].value_counts())

# Create a label mapping after standardization
system_label_mapping = {system: idx for idx, system in enumerate(balanced_data['System'].unique())}
balanced_data['labels'] = balanced_data['System'].map(system_label_mapping)
print("Updated System label mapping:\n", system_label_mapping)
joblib.dump(system_label_mapping, 'system_label_mapping_balanced.pkl')


# Convert to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(balanced_data)
split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
print("Dataset split into train and test sets.\nTrain set size:", len(split_dataset['train']), "\nTest set size:", len(split_dataset['test']))

# Tokenizer and model preparation
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(system_label_mapping))
print("Model and tokenizer initialized.")

# Tokenization function
def tokenize_function(examples):
    tokens = tokenizer(examples['Keywords for n=3'], padding="max_length", truncation=True, max_length=64)
    tokens['labels'] = examples['labels']
    return tokens

# Tokenize dataset
tokenized_dataset = split_dataset.map(tokenize_function, batched=True)
print("Tokenization complete. Example tokenized data:\n", tokenized_dataset['train'][0])

# Compute class weights for balanced training
class_weights = compute_class_weight('balanced', classes=np.unique(balanced_data['System']), y=balanced_data['System'])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print("Class weights computed:\n", class_weights)

# Display initial model parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")  # Print first 2 values for brevity

# Metric computation function
def compute_metrics_function(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Training arguments with frequent evaluation steps
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",  # Evaluate at specific step intervals
    eval_steps=500,               # Evaluate every 500 steps to show accuracy
    logging_steps=100,             # Log progress every 100 steps
    learning_rate=1e-4,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    logging_dir='./logs'
)

# Custom Trainer with class weights
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Initialize and train model
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics_function,
)

# Print a summary of the training configuration
print("Training configuration:\n", trainer.args)

# Start training
trainer.train()

# Evaluate and print results
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Save model and tokenizer
trainer.save_model('distilbert_system_classifier_balanced')
tokenizer.save_pretrained('distilbert_system_classifier_balanced')
