import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import re

class MinimalEmailSpamNaiveBayes:
    def __init__(self, csv_file="E:\\Baki\\final_email_dataset_cleaned.csv"):
        self.csv_file = csv_file
        self.model = None
        self.feature_names = ['char_count', 'word_count', 'sentence_count']
        self.X_train = None
        self.y_train = None
        
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
        """Train the Naive Bayes model"""
        print("🧠 Training Naive Bayes...")
        
        # Load data
        X_text, y = self.load_data()
        
        # Create features
        X_features = self.create_basic_features(X_text)
        
        print(f"📊 Feature statistics:")
        print(X_features.describe())
        
        # Split data
        self.X_train, X_test, self.y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data split:")
        print(f"   Training: {len(self.X_train):,} emails")
        print(f"   Testing: {len(X_test):,} emails")
        
        # Create and train Naive Bayes model
        self.model = GaussianNB()
        
        print(f"🧠 Training Gaussian Naive Bayes...")
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Training completed!")
        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show detailed results
        self.show_results(y_test, y_pred, y_pred_proba, X_test)
        
        return self.model
    
    def show_results(self, y_test, y_pred, y_pred_proba, X_test):
        """Display results and visualizations"""
        
        # Print detailed metrics
        print(f"\n📊 DETAILED RESULTS:")
        print("="*50)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))
        
        # Show class statistics from training data
        self.show_class_statistics()
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axes[0,0])
        axes[0,0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # 2. Class Prior Probabilities
        class_priors = [np.mean(self.y_train == 0), np.mean(self.y_train == 1)]
        axes[0,1].bar(['Ham (0)', 'Spam (1)'], class_priors, color=['green', 'red'], alpha=0.7)
        axes[0,1].set_title('Class Prior Probabilities\n(from training data)')
        axes[0,1].set_ylabel('Probability')
        
        # Add value labels
        for i, v in enumerate(class_priors):
            axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Feature Means by Class
        ham_data = self.X_train[self.y_train == 0]
        spam_data = self.X_train[self.y_train == 1]
        
        ham_means = ham_data.mean()
        spam_means = spam_data.mean()
        
        x = np.arange(len(self.feature_names))
        width = 0.35
        
        bars1 = axes[0,2].bar(x - width/2, ham_means, width, label='Ham', color='green', alpha=0.7)
        bars2 = axes[0,2].bar(x + width/2, spam_means, width, label='Spam', color='red', alpha=0.7)
        
        axes[0,2].set_title('Feature Means by Class\n(Naive Bayes Assumptions)')
        axes[0,2].set_xlabel('Features')
        axes[0,2].set_ylabel('Mean Value')
        axes[0,2].set_xticks(x)
        axes[0,2].set_xticklabels(['Chars', 'Words', 'Sentences'])
        axes[0,2].legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[0,2].text(bar.get_x() + bar.get_width()/2., height + height*0.01, 
                          f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            axes[0,2].text(bar.get_x() + bar.get_width()/2., height + height*0.01, 
                          f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Feature Distributions (Character Count)
        axes[1,0].hist(ham_data['char_count'], bins=50, alpha=0.7, label='Ham', 
                      color='green', density=True, range=(0, 5000))
        axes[1,0].hist(spam_data['char_count'], bins=50, alpha=0.7, label='Spam', 
                      color='red', density=True, range=(0, 5000))
        axes[1,0].set_title('Character Count Distribution\n(Naive Bayes assumes Gaussian)')
        axes[1,0].set_xlabel('Characters')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        
        # 5. Feature Distributions (Word Count)
        axes[1,1].hist(ham_data['word_count'], bins=50, alpha=0.7, label='Ham', 
                      color='green', density=True, range=(0, 800))
        axes[1,1].hist(spam_data['word_count'], bins=50, alpha=0.7, label='Spam', 
                      color='red', density=True, range=(0, 800))
        axes[1,1].set_title('Word Count Distribution\n(Gaussian Assumption)')
        axes[1,1].set_xlabel('Words')
        axes[1,1].set_ylabel('Density')
        axes[1,1].legend()
        
        # 6. Prediction Confidence Distribution
        confidence = np.max(y_pred_proba, axis=1)
        axes[1,2].hist(confidence, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1,2].set_title('Prediction Confidence Distribution')
        axes[1,2].set_xlabel('Confidence (Max Probability)')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].axvline(confidence.mean(), color='red', linestyle='--', 
                         label=f'Mean: {confidence.mean():.3f}')
        axes[1,2].legend()
        
        plt.tight_layout()
        
        # Save plot
        output_path = "E:\\Baki\\naive_bayes_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Results saved: {output_path}")
        plt.show()
        
        # Print detailed feature statistics
        print(f"\n📊 NAIVE BAYES FEATURE STATISTICS:")
        print("="*70)
        print(f"{'Feature':<15} {'Ham Mean':<10} {'Ham Std':<10} {'Spam Mean':<11} {'Spam Std':<10} {'Difference':<10}")
        print("-" * 76)
        
        for feature in self.feature_names:
            ham_mean = ham_data[feature].mean()
            ham_std = ham_data[feature].std()
            spam_mean = spam_data[feature].mean()
            spam_std = spam_data[feature].std()
            difference = ham_mean - spam_mean
            
            print(f"{feature:<15} {ham_mean:<10.1f} {ham_std:<10.1f} {spam_mean:<11.1f} {spam_std:<10.1f} {difference:<10.1f}")
    
    def show_class_statistics(self):
        """Show the learned statistics from Naive Bayes"""
        print(f"\n🧠 NAIVE BAYES LEARNED PARAMETERS:")
        print("="*50)
        
        # Class priors
        ham_prior = np.mean(self.y_train == 0)
        spam_prior = np.mean(self.y_train == 1)
        print(f"Class Priors:")
        print(f"   P(Ham) = {ham_prior:.4f} ({ham_prior*100:.1f}%)")
        print(f"   P(Spam) = {spam_prior:.4f} ({spam_prior*100:.1f}%)")
        
        # Feature means and variances per class
        print(f"\nFeature Statistics (Gaussian assumptions):")
        
        ham_data = self.X_train[self.y_train == 0]
        spam_data = self.X_train[self.y_train == 1]
        
        for i, feature in enumerate(self.feature_names):
            print(f"\n{feature.replace('_', ' ').title()}:")
            ham_mean = ham_data[feature].mean()
            ham_var = ham_data[feature].var()
            spam_mean = spam_data[feature].mean()
            spam_var = spam_data[feature].var()
            
            print(f"   Ham:  μ = {ham_mean:.1f}, σ² = {ham_var:.1f}")
            print(f"   Spam: μ = {spam_mean:.1f}, σ² = {spam_var:.1f}")
    
    def predict_single(self, text):
        """Predict if a single email is spam or ham with detailed probabilities"""
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
        probabilities = self.model.predict_proba(features)[0]
        
        result = "Spam" if prediction == 1 else "Ham"
        confidence = max(probabilities) * 100
        
        print(f"\n📧 NAIVE BAYES EMAIL ANALYSIS:")
        print("="*40)
        print(f"Email Features:")
        print(f"   Characters: {features['char_count'][0]}")
        print(f"   Words: {features['word_count'][0]}")
        print(f"   Sentences: {features['sentence_count'][0]}")
        
        print(f"\nProbability Calculations:")
        print(f"   P(Ham|features) = {probabilities[0]:.4f} ({probabilities[0]*100:.1f}%)")
        print(f"   P(Spam|features) = {probabilities[1]:.4f} ({probabilities[1]*100:.1f}%)")
        
        print(f"\nFinal Classification: {result}")
        print(f"Confidence: {confidence:.1f}%")
        
        # Show which class this email is most similar to
        ham_data = self.X_train[self.y_train == 0]
        spam_data = self.X_train[self.y_train == 1]
        
        # Calculate how "typical" this email is for each class
        ham_mean = ham_data.mean()
        spam_mean = spam_data.mean()
        
        print(f"\nComparison to typical emails:")
        print(f"   Typical Ham:  {ham_mean['char_count']:.0f} chars, {ham_mean['word_count']:.0f} words, {ham_mean['sentence_count']:.1f} sentences")
        print(f"   Typical Spam: {spam_mean['char_count']:.0f} chars, {spam_mean['word_count']:.0f} words, {spam_mean['sentence_count']:.1f} sentences")
        print(f"   This email:   {features['char_count'][0]} chars, {features['word_count'][0]} words, {features['sentence_count'][0]} sentences")
        
        return prediction
    
    def explain_naive_bayes(self):
        """Explain how Naive Bayes works with these features"""
        print(f"\n🧠 HOW NAIVE BAYES WORKS:")
        print("="*50)
        print("Naive Bayes uses Bayes' theorem with a 'naive' independence assumption.")
        print("\nFor each email, it calculates:")
        print("   P(Spam|chars,words,sentences) ∝ P(Spam) × P(chars|Spam) × P(words|Spam) × P(sentences|Spam)")
        print("   P(Ham|chars,words,sentences)  ∝ P(Ham)  × P(chars|Ham)  × P(words|Ham)  × P(sentences|Ham)")
        
        print(f"\nThe 'naive' assumption: features are independent given the class")
        print(f"   (i.e., knowing word count doesn't tell us about character count)")
        
        print(f"\nGaussian Naive Bayes assumes each feature follows a normal distribution")
        print(f"within each class (Ham/Spam).")
        
        if self.model is not None:
            print(f"\nYour model learned:")
            ham_prior = np.mean(self.y_train == 0)
            spam_prior = np.mean(self.y_train == 1)
            print(f"   • {ham_prior:.1%} of emails are Ham")
            print(f"   • {spam_prior:.1%} of emails are Spam")
            print(f"   • Different means/variances for each feature in Ham vs Spam emails")

def run_minimal_naive_bayes_detection():
    """Run the minimal spam detection with Naive Bayes"""
    print("🧠 MINIMAL EMAIL SPAM DETECTION - NAIVE BAYES")
    print("Using only 3 features: Character count, Word count, Sentence count")
    print("="*70)
    
    # Initialize and train
    classifier = MinimalEmailSpamNaiveBayes()
    model = classifier.train()
    
    # Explain how it works
    classifier.explain_naive_bayes()
    
    print(f"\n✅ Naive Bayes model ready!")
    print(f"📊 Features used: Only 3 basic length metrics")
    print(f"🧠 Model: Gaussian Naive Bayes with independence assumption")
    print(f"📁 Results visualization saved")
    
    return classifier

if __name__ == "__main__":
    # Run the Naive Bayes version
    spam_detector = run_minimal_naive_bayes_detection()
    
    # Optional: Test with sample emails
    print(f"\n🧪 Test with sample emails? (y/n): ", end="")
    try:
        test_choice = input().strip().lower()
        if test_choice in ['y', 'yes']:
            
            print(f"\n--- Testing Sample Emails ---")
            
            # Test with a short spam-like text
            spam_sample = "WIN $1000 NOW! Click here!"
            print(f"\nSample 1 (likely spam): '{spam_sample}'")
            spam_detector.predict_single(spam_sample)
            
            # Test with a longer ham-like text
            ham_sample = "Hi John, I hope you're doing well. I wanted to follow up on our meeting last week regarding the project timeline. Could we schedule another meeting to discuss the next steps? Let me know what works for your schedule. Best regards, Sarah"
            print(f"\nSample 2 (likely ham): '{ham_sample[:60]}...'")
            spam_detector.predict_single(ham_sample)
            
            # Test with custom email
            print(f"\nEnter your own email text to test:")
            custom_text = input("Email text: ")
            if custom_text.strip():
                spam_detector.predict_single(custom_text)
            
    except:
        pass
    
    print(f"\n👋 Done!")