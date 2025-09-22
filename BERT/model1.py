import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  # Import AdamW from torch instead
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the dataset
df = pd.read_csv("E:\\Baki\\final_email_dataset_cleaned.csv")
print(f"Dataset shape: {df.shape}")
print(f"Dataset info:")
print(df.info())
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# Custom Dataset class
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Preprocessing
print("Preprocessing data...")
df['text'] = df['text'].fillna('')  # Handle any NaN values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

# Further split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Initialize BERT tokenizer and model
MODEL_NAME = 'distilbert-base-uncased'
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)
# Move model to device
model = model.to(device)

# Create datasets
train_dataset = EmailDataset(X_train, y_train, tokenizer)
val_dataset = EmailDataset(X_val, y_val, tokenizer)
test_dataset = EmailDataset(X_test, y_test, tokenizer)

# Create data loaders
BATCH_SIZE = 32
# Add these parameters to your DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=0,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training parameters
EPOCHS = 3
LEARNING_RATE = 2e-5

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training function
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for data in tqdm(data_loader, desc="Training"):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Evaluation function
def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return (correct_predictions.double() / len(data_loader.dataset), 
            np.mean(losses), predictions, actual_labels)

# Training loop
print("Starting training...")
best_accuracy = 0
train_accuracies = []
val_accuracies = []

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    print('-' * 50)
    
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
    print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')
    
    val_acc, val_loss, _, _ = eval_model(model, val_loader, device)
    print(f'Val loss {val_loss:.4f} accuracy {val_acc:.4f}')
    
    train_accuracies.append(train_acc.item())
    val_accuracies.append(val_acc.item())
    
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_spam_classifier.pth')
        best_accuracy = val_acc

# Load best model and evaluate on test set
model.load_state_dict(torch.load('best_spam_classifier.pth'))
test_acc, test_loss, test_predictions, test_labels = eval_model(model, test_loader, device)

print(f'\nTest Results:')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_predictions, 
                          target_names=['Ham (0)', 'Spam (1)']))

# Confusion Matrix
cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Function to predict single email
def predict_email(text, model, tokenizer, device, max_length=512):
    model.eval()
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        return predicted_class.item(), confidence.item()

# Example usage
sample_email = "Congratulations! You've won $1000! Click here to claim your prize now!"
prediction, confidence = predict_email(sample_email, model, tokenizer, device)
print(f"\nSample Prediction:")
print(f"Email: {sample_email}")
print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")
print(f"Confidence: {confidence:.4f}")