import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC  # Changed from SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import seaborn as sns
import re

class MinimalEmailSpamSVM:
    def __init__(self, csv_file="E:\\Baki\\final_email_dataset_cleaned.csv"):
        self.csv_file = csv_file
        self.model = None
        
    def load_data(self):
        """Load the dataset"""
        print("📊 Loading data...")
        df = pd.read_csv(self.csv_file)
        print(f"✅ Loaded {len(df):,} emails")
        return df['text'], df['label']
    
    def create_basic_features(self, texts):
        """Create only 3 basic features"""
        print("🔧 Creating basic features...")
        
        features = pd.DataFrame()
        
        # 1. Character count
        features['char_count'] = texts.str.len()
        
        # 2. Word count
        features['word_count'] = texts.str.split().str.len()
        
        # 3. Sentence count
        features['sentence_count'] = texts.apply(lambda x: len(re.split(r'[.!?]+', x)))
        
        print(f"✅ Created 3 features: char_count, word_count, sentence_count")
        return features
    
    def train(self):
        """Train the minimal SVM model"""
        print("🤖 Training minimal LinearSVM...")
        
        # Load data
        X_text, y = self.load_data()
        
        # Create features
        X_features = self.create_basic_features(X_text)
        
        print(f"📊 Feature statistics:")
        print(X_features.describe())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data split:")
        print(f"   Training: {len(X_train):,} emails")
        print(f"   Testing: {len(X_test):,} emails")
        
        # Create and train model with LinearSVC
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', LinearSVC(
                C=1.0, 
                random_state=42,
                max_iter=2000,  # Added to prevent convergence warnings
                dual=False,     # Recommended when n_samples > n_features
                class_weight='balanced'  # Handle potential class imbalance
            ))
        ])
        
        print(f"⚙️  Using LinearSVM parameters:")
        print(f"   • C: 1.0 (regularization)")
        print(f"   • Max iterations: 2000")
        print(f"   • Dual: False (faster for this dataset size)")
        print(f"   • Class weight: balanced")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Training completed!")
        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show detailed results
        self.show_results(y_test, y_pred, X_train, X_test, y_train)
        
        return self.model
    
    def show_results(self, y_test, y_pred, X_train, X_test, y_train):
        """Display results and visualizations"""
        
        # Print detailed metrics
        print(f"\n📊 DETAILED RESULTS:")
        print("="*50)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Update title to reflect LinearSVC
        fig.suptitle('Minimal LinearSVM Spam Classification Results', fontsize=14, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axes[0,0])
        axes[0,0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # 2. Feature Distribution by Class
        # Combine train and test for full picture
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([pd.Series(y_train), pd.Series(y_test)])
        
        ham_data = X_full[y_full == 0]
        spam_data = X_full[y_full == 1]
        
        # Character count distribution
        axes[0,1].hist(ham_data['char_count'], bins=50, alpha=0.7, label='Ham', color='green', density=True)
        axes[0,1].hist(spam_data['char_count'], bins=50, alpha=0.7, label='Spam', color='red', density=True)
        axes[0,1].set_xlim(0, 5000)  # Limit for better visualization
        axes[0,1].set_title('Character Count Distribution')
        axes[0,1].set_xlabel('Characters')
        axes[0,1].set_ylabel('Density')
        axes[0,1].legend()
        
        # 3. Word count distribution
        axes[1,0].hist(ham_data['word_count'], bins=50, alpha=0.7, label='Ham', color='green', density=True)
        axes[1,0].hist(spam_data['word_count'], bins=50, alpha=0.7, label='Spam', color='red', density=True)
        axes[1,0].set_xlim(0, 1000)  # Limit for better visualization
        axes[1,0].set_title('Word Count Distribution')
        axes[1,0].set_xlabel('Words')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        
        # 4. Average feature comparison
        ham_avg = ham_data.mean()
        spam_avg = spam_data.mean()
        
        features = ['Characters', 'Words', 'Sentences']
        ham_values = [ham_avg['char_count'], ham_avg['word_count'], ham_avg['sentence_count']]
        spam_values = [spam_avg['char_count'], spam_avg['word_count'], spam_avg['sentence_count']]
        
        x = np.arange(len(features))
        width = 0.35
        
        axes[1,1].bar(x - width/2, ham_values, width, label='Ham', color='green', alpha=0.7)
        axes[1,1].bar(x + width/2, spam_values, width, label='Spam', color='red', alpha=0.7)
        
        axes[1,1].set_title('Average Feature Values')
        axes[1,1].set_xlabel('Features')
        axes[1,1].set_ylabel('Average Count')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(features)
        axes[1,1].legend()
        
        # Add value labels on bars
        for i, (ham_val, spam_val) in enumerate(zip(ham_values, spam_values)):
            axes[1,1].text(i - width/2, ham_val + max(ham_values)*0.01, f'{ham_val:.0f}', 
                          ha='center', va='bottom', fontsize=9)
            axes[1,1].text(i + width/2, spam_val + max(spam_values)*0.01, f'{spam_val:.0f}', 
                          ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot with updated name
        output_path = "E:\\Baki\\minimal_linearsvm_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Results saved: {output_path}")
        plt.show()
        
        # Print feature comparison
        print(f"\n📊 FEATURE COMPARISON:")
        print("="*50)
        print(f"{'Feature':<15} {'Ham Average':<12} {'Spam Average':<13} {'Difference':<12}")
        print("-" * 52)
        print(f"{'Characters':<15} {ham_avg['char_count']:<12.0f} {spam_avg['char_count']:<13.0f} {ham_avg['char_count'] - spam_avg['char_count']:<12.0f}")
        print(f"{'Words':<15} {ham_avg['word_count']:<12.0f} {spam_avg['word_count']:<13.0f} {ham_avg['word_count'] - spam_avg['word_count']:<12.0f}")
        print(f"{'Sentences':<15} {ham_avg['sentence_count']:<12.1f} {spam_avg['sentence_count']:<13.1f} {ham_avg['sentence_count'] - spam_avg['sentence_count']:<12.1f}")
        
        # Display LinearSVC-specific information
        print(f"\n🔧 LINEARSVM MODEL INFO:")
        print("="*30)
        print(f"Algorithm: LinearSVC (liblinear)")
        print(f"Optimization: Faster than SVC for linear problems")
        print(f"Memory: More efficient for large datasets")
        print(f"Dual formulation: {self.model.named_steps['svm'].dual}")
        print(f"Max iterations: {self.model.named_steps['svm'].max_iter}")
    
    def predict_single(self, text):
        """Predict if a single email is spam or ham"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        # Create features for single text
        features = pd.DataFrame({
            'char_count': [len(text)],
            'word_count': [len(text.split())],
            'sentence_count': [len(re.split(r'[.!?]+', text))]
        })
        
        prediction = self.model.predict(features)[0]
        result = "Spam" if prediction == 1 else "Ham"
        
        print(f"\n📧 Email Analysis (LinearSVM):")
        print(f"   Characters: {features['char_count'][0]}")
        print(f"   Words: {features['word_count'][0]}")
        print(f"   Sentences: {features['sentence_count'][0]}")
        print(f"   Classification: {result}")
        
        return prediction

def run_minimal_spam_detection():
    """Run the minimal spam detection with only 3 features using LinearSVC"""
    print("🚀 MINIMAL EMAIL SPAM DETECTION - LinearSVM")
    print("Using LinearSVC instead of SVC(kernel='linear')")
    print("Features: Character count, Word count, Sentence count")
    print("Benefits: Faster training, more memory efficient")
    print("="*60)
    
    # Initialize and train
    classifier = MinimalEmailSpamSVM()
    model = classifier.train()
    
    print(f"\n✅ Minimal LinearSVM model ready!")
    print(f"📊 Features used: Only 3 basic length metrics")
    print(f"⚡ Algorithm: LinearSVC (optimized for linear problems)")
    print(f"📁 Results visualization saved")
    
    return classifier

if __name__ == "__main__":
    # Run the minimal version with LinearSVC
    spam_detector = run_minimal_spam_detection()
    
    # Optional: Test with sample emails
    print(f"\n🧪 Test with sample emails? (y/n): ", end="")
    try:
        test_choice = input().strip().lower()
        if test_choice in ['y', 'yes']:
            
            # Test with a short spam-like text
            print(f"\n--- Testing Sample Emails with LinearSVM ---")
            
            spam_sample = "WIN $1000 NOW! Click here!"
            print(f"\nSample 1 (likely spam): '{spam_sample}'")
            spam_detector.predict_single(spam_sample)
            
            ham_sample = "Hi John, I hope you're doing well. I wanted to follow up on our meeting last week regarding the project timeline. Could we schedule another meeting to discuss the next steps? Let me know what works for your schedule. Best regards, Sarah"
            print(f"\nSample 2 (likely ham): '{ham_sample[:60]}...'")
            spam_detector.predict_single(ham_sample)
            
    except:
        pass
    
    print(f"\n👋 Done!")