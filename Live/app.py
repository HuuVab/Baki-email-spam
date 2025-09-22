from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

# Global variables for model
model = None
tokenizer = None
device = None

# Simple in-memory storage for emails
sent_emails = []
received_emails = []

def load_trained_model(model_path='best_spam_classifier.pth'):
    """Load the trained spam classifier model"""
    global model, tokenizer, device
    
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
    
    print("Model loaded successfully!")

def classify_email(text):
    """Classify a single email as spam or ham"""
    if model is None:
        return {'error': 'Model not loaded'}
    
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

# Add some sample received emails for demo
def initialize_sample_emails():
    sample_emails = [
        {
            'from': 'friend@example.com',
            'subject': 'Meeting tomorrow',
            'body': 'Hi! Are we still meeting tomorrow for lunch? Let me know if you need to reschedule.',
            'timestamp': '2024-01-15 10:30:00'
        },
        {
            'from': 'winner@lottery.com',
            'subject': 'CONGRATULATIONS! You won $1,000,000!!!',
            'body': 'You have won the international lottery! Send us your bank details immediately to claim your prize! Act now, limited time offer!',
            'timestamp': '2024-01-15 09:15:00'
        },
        {
            'from': 'boss@company.com',
            'subject': 'Project Update Required',
            'body': 'Please send me the latest project status report by end of day. Thanks!',
            'timestamp': '2024-01-15 08:45:00'
        },
        {
            'from': 'deals@shopping.com',
            'subject': 'FREE MONEY! CLICK NOW! URGENT!!!',
            'body': 'Make money fast! No work required! Click this link now and earn thousands! Limited time only!',
            'timestamp': '2024-01-15 07:30:00'
        }
    ]
    
    for email_data in sample_emails:
        classification = classify_email(f"{email_data['subject']} {email_data['body']}")
        email_data.update({
            'id': str(uuid.uuid4()),
            'classification': classification,
            'read': False
        })
        received_emails.append(email_data)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify_text():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        result = classify_email(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/send-email', methods=['POST'])
def send_email():
    try:
        data = request.json
        to_email = data.get('to_email')
        subject = data.get('subject')
        body = data.get('body')
        
        if not all([to_email, subject, body]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Simulate sending email by storing it
        email_data = {
            'id': str(uuid.uuid4()),
            'to': to_email,
            'subject': subject,
            'body': body,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'sent'
        }
        
        # Classify the sent email
        classification = classify_email(f"{subject} {body}")
        email_data['classification'] = classification
        
        sent_emails.append(email_data)
        
        return jsonify({
            'message': 'Email sent successfully (simulated)',
            'email_data': email_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/inbox', methods=['GET'])
def get_inbox():
    try:
        # Return received emails sorted by timestamp (newest first)
        sorted_emails = sorted(received_emails, key=lambda x: x['timestamp'], reverse=True)
        return jsonify({'emails': sorted_emails})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sent', methods=['GET'])
def get_sent():
    try:
        # Return sent emails sorted by timestamp (newest first)
        sorted_emails = sorted(sent_emails, key=lambda x: x['timestamp'], reverse=True)
        return jsonify({'emails': sorted_emails})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate-receive', methods=['POST'])
def simulate_receive():
    try:
        data = request.json
        from_email = data.get('from_email', 'unknown@example.com')
        subject = data.get('subject')
        body = data.get('body')
        
        if not all([subject, body]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Simulate receiving an email
        classification = classify_email(f"{subject} {body}")
        
        email_data = {
            'id': str(uuid.uuid4()),
            'from': from_email,
            'subject': subject,
            'body': body,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'classification': classification,
            'read': False
        }
        
        received_emails.append(email_data)
        
        return jsonify({
            'message': 'Email received successfully (simulated)',
            'email_data': email_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mark-read/<email_id>', methods=['POST'])
def mark_read(email_id):
    try:
        for email in received_emails:
            if email['id'] == email_id:
                email['read'] = True
                return jsonify({'message': 'Email marked as read'})
        
        return jsonify({'error': 'Email not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load the model on startup
    try:
        load_trained_model()
        initialize_sample_emails()
        print("Application started with sample emails!")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)