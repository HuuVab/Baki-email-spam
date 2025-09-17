import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import re

class MinimalEmailSpamRandomForest:
    def __init__(self, csv_file="E:\\Baki\\final_email_dataset_cleaned.csv"):
        self.csv_file = csv_file
        self.model = None
        self.feature_names = ['char_count', 'word_count', 'sentence_count']
        
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
        """Train the Random Forest model"""
        print("🌲 Training Random Forest...")
        
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
        
        # Create and train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,        # Number of trees
            max_depth=10,           # Maximum depth of trees
            min_samples_split=5,    # Minimum samples to split
            min_samples_leaf=2,     # Minimum samples in leaf
            random_state=42,        # For reproducibility
            n_jobs=-1               # Use all cores
        )
        
        print(f"🌲 Training Random Forest with {self.model.n_estimators} trees...")
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
        
        # Feature importance from Random Forest
        feature_importance = self.model.feature_importances_
        print(f"\n🌲 FEATURE IMPORTANCE:")
        print("="*30)
        for i, (feature, importance) in enumerate(zip(self.feature_names, feature_importance)):
            print(f"{feature:<15}: {importance:.4f} ({importance*100:.1f}%)")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axes[0,0])
        axes[0,0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # 2. Feature Importance
        axes[0,1].bar(self.feature_names, feature_importance, color='forestgreen', alpha=0.7)
        axes[0,1].set_title('Random Forest Feature Importance')
        axes[0,1].set_ylabel('Importance Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(feature_importance):
            axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Feature Distribution by Class
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([pd.Series(y_train), pd.Series(y_test)])
        
        ham_data = X_full[y_full == 0]
        spam_data = X_full[y_full == 1]
        
        # Character count distribution
        axes[1,0].hist(ham_data['char_count'], bins=50, alpha=0.7, label='Ham', color='green', density=True)
        axes[1,0].hist(spam_data['char_count'], bins=50, alpha=0.7, label='Spam', color='red', density=True)
        axes[1,0].set_xlim(0, 5000)  # Limit for better visualization
        axes[1,0].set_title('Character Count Distribution')
        axes[1,0].set_xlabel('Characters')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        
        # 4. Average feature comparison with Random Forest prediction boundaries
        ham_avg = ham_data.mean()
        spam_avg = spam_data.mean()
        
        features = ['Characters', 'Words', 'Sentences']
        ham_values = [ham_avg['char_count'], ham_avg['word_count'], ham_avg['sentence_count']]
        spam_values = [spam_avg['char_count'], spam_avg['word_count'], spam_avg['sentence_count']]
        
        x = np.arange(len(features))
        width = 0.35
        
        bars1 = axes[1,1].bar(x - width/2, ham_values, width, label='Ham', color='green', alpha=0.7)
        bars2 = axes[1,1].bar(x + width/2, spam_values, width, label='Spam', color='red', alpha=0.7)
        
        axes[1,1].set_title('Average Feature Values')
        axes[1,1].set_xlabel('Features')
        axes[1,1].set_ylabel('Average Count')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(features)
        axes[1,1].legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + height*0.01, 
                          f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + height*0.01, 
                          f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        output_path = "E:\\Baki\\random_forest_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Results saved: {output_path}")
        plt.show()
        
        # Print detailed feature comparison
        print(f"\n📊 FEATURE COMPARISON:")
        print("="*60)
        print(f"{'Feature':<15} {'Ham Average':<12} {'Spam Average':<13} {'Difference':<12} {'RF Importance':<12}")
        print("-" * 64)
        for i, feature in enumerate(['Characters', 'Words', 'Sentences']):
            ham_val = ham_values[i]
            spam_val = spam_values[i]
            diff = ham_val - spam_val
            importance = feature_importance[i]
            print(f"{feature:<15} {ham_val:<12.0f} {spam_val:<13.0f} {diff:<12.0f} {importance:<12.3f}")
    
    def predict_single(self, text):
        """Predict if a single email is spam or ham with probability"""
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
        probability = self.model.predict_proba(features)[0]
        
        result = "Spam" if prediction == 1 else "Ham"
        confidence = max(probability) * 100
        
        print(f"\n📧 Email Analysis:")
        print(f"   Characters: {features['char_count'][0]}")
        print(f"   Words: {features['word_count'][0]}")
        print(f"   Sentences: {features['sentence_count'][0]}")
        print(f"   Classification: {result}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Probabilities: Ham={probability[0]:.3f}, Spam={probability[1]:.3f}")
        
        return prediction
    
    def show_tree_info(self):
        """Show information about the Random Forest trees"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return
        
        print(f"\n🌲 RANDOM FOREST MODEL INFO:")
        print("="*40)
        print(f"Number of trees: {self.model.n_estimators}")
        print(f"Max depth: {self.model.max_depth}")
        print(f"Min samples split: {self.model.min_samples_split}")
        print(f"Min samples leaf: {self.model.min_samples_leaf}")
        print(f"Features used: {len(self.feature_names)}")
        print(f"Classes: {self.model.classes_}")
        
        # Show some tree depths
        tree_depths = [tree.get_depth() for tree in self.model.estimators_[:10]]
        print(f"Sample tree depths (first 10): {tree_depths}")

def run_minimal_random_forest_detection():
    """Run the minimal spam detection with Random Forest"""
    print("🌲 MINIMAL EMAIL SPAM DETECTION - RANDOM FOREST")
    print("Using only 3 features: Character count, Word count, Sentence count")
    print("="*70)
    
    # Initialize and train
    classifier = MinimalEmailSpamRandomForest()
    model = classifier.train()
    
    # Show model info
    classifier.show_tree_info()
    
    print(f"\n✅ Random Forest model ready!")
    print(f"📊 Features used: Only 3 basic length metrics")
    print(f"🌲 Model: 100 decision trees")
    print(f"📁 Results visualization saved")
    
    return classifier

if __name__ == "__main__":
    # Run the Random Forest version
    spam_detector = run_minimal_random_forest_detection()
    
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