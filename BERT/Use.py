import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def load_trained_model(model_path='best_spam_classifier.pth'):
    """Load the trained spam classifier model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Recreate model architecture
    MODEL_NAME = 'distilbert-base-uncased'
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    return model, tokenizer, device

def classify_email(text, model, tokenizer, device):
    """Classify a single email as spam or ham"""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
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
        probabilities = F.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        return {
            'prediction': 'Spam' if predicted_class.item() == 1 else 'Ham',
            'confidence': confidence.item(),
            'spam_probability': probabilities[0][1].item(),
            'ham_probability': probabilities[0][0].item()
        }

# Usage
if __name__ == "__main__":
    # Load the model
    model, tokenizer, device = load_trained_model()
    
    # Test with new emails
    test_email = "Congratulations! You've won $1000! Click here to claim your prize now!"
    result = classify_email(test_email, model, tokenizer, device)
    
    print(f"Email: {test_email}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Spam Probability: {result['spam_probability']:.4f}")
    print(f"Ham Probability: {result['ham_probability']:.4f}")