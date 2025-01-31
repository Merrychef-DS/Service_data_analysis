import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
data = pd.read_csv("Processed_FSR_Solution_Keywords.csv")

system_keywords_df = data.groupby('System')['FSR_Solution'].apply(lambda x: ' '.join(x.dropna())).reset_index()
system_keywords_df['System'] = system_keywords_df['System'].str.lower()
system_keywords_df = system_keywords_df[system_keywords_df['FSR_Solution'].str.strip().astype(bool)]
system_keywords_df = system_keywords_df.dropna(subset=['System'])

system_label_mapping = {system: idx for idx, system in enumerate(system_keywords_df['System'].unique())}
system_keywords_df['labels'] = system_keywords_df['System'].map(system_label_mapping)

output_file = 'system_keywords_dataset.csv'
system_keywords_df.to_csv(output_file, index=False)

hf_dataset = Dataset.from_pandas(system_keywords_df)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(system_keywords_df['System'].unique()))

def tokenize_function(examples):
    tokens = tokenizer(examples['FSR_Solution'], padding="max_length", truncation=True)
    tokens['labels'] = examples['labels']
    return tokens

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
h
import torch.nn as nn
import json

class_labels = system_keywords_df['System'].unique()
y_bert = system_keywords_df['System'].str.strip().str.lower()
num_classes = len(system_label_mapping)
print("Classes from dataset: ", system_keywords_df['System'].unique())
print("Classes used for weights: ", class_labels)
# Recompute class weights ensuring all classes are included
class_weights = compute_class_weight('balanced', classes=class_labels, y=system_keywords_df['System'])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print("Number of classes: ", num_classes)
print("Class weights calculated for: ", len(class_weights_tensor))
# Ensure the weights tensor matches the number of classes
if len(class_weights_tensor) != num_classes:
    raise ValueError("Mismatch in class weights and number of classes. Expected {}, got {}".format(num_classes, len(class_weights_tensor)))


def compute_metrics_function(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,
    weight_decay=0.01,
    logging_dir='./logs',
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    compute_metrics=compute_metrics_function,
)

trainer.train()
with open('system_label_mapping.json', 'w') as f:
    json.dump(system_label_mapping, f)
model_save_path = 'distilbert_system_classifier'
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)