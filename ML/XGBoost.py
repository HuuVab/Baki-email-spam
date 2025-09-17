import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as xgb
import seaborn as sns
import re

class MinimalEmailSpamXGBoost:
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
        """Train the minimal XGBoost model"""
        print("🚀 Training minimal XGBoost...")
        
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
        
        # Create and train XGBoost model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                verbosity=0  # Reduce output noise
            ))
        ])
        
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
        
        # Get feature importance from XGBoost
        xgb_model = self.model.named_steps['xgb']
        feature_names = ['char_count', 'word_count', 'sentence_count']
        feature_importance = xgb_model.feature_importances_
        
        print(f"\n🔍 FEATURE IMPORTANCE (XGBoost):")
        print("="*40)
        for name, importance in zip(feature_names, feature_importance):
            print(f"{name:<15}: {importance:.4f} ({importance*100:.2f}%)")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axes[0,0])
        axes[0,0].set_title(f'Confusion Matrix (XGBoost)\nAccuracy: {accuracy:.4f}')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # 2. Feature Importance
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        bars = axes[0,1].bar(feature_names, feature_importance, color=colors, alpha=0.7)
        axes[0,1].set_title('XGBoost Feature Importance')
        axes[0,1].set_ylabel('Importance Score')
        axes[0,1].set_xlabel('Features')
        
        # Add value labels on bars
        for bar, importance in zip(bars, feature_importance):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Feature Distribution by Class
        # Combine train and test for full picture
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
        
        # Save plot
        output_path = "E:\\Baki\\minimal_xgboost_results.png"
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
        probability = self.model.predict_proba(features)[0]
        
        result = "Spam" if prediction == 1 else "Ham"
        confidence = max(probability) * 100
        
        print(f"\n📧 Email Analysis (XGBoost):")
        print(f"   Characters: {features['char_count'][0]}")
        print(f"   Words: {features['word_count'][0]}")
        print(f"   Sentences: {features['sentence_count'][0]}")
        print(f"   Classification: {result}")
        print(f"   Confidence: {confidence:.2f}%")
        print(f"   Probabilities: Ham={probability[0]:.4f}, Spam={probability[1]:.4f}")
        
        return prediction
    
    def get_feature_importance(self):
        """Get and display feature importance"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        xgb_model = self.model.named_steps['xgb']
        feature_names = ['char_count', 'word_count', 'sentence_count']
        importance_dict = dict(zip(feature_names, xgb_model.feature_importances_))
        
        return importance_dict

def run_minimal_spam_detection():
    """Run the minimal spam detection with XGBoost"""
    print("🚀 MINIMAL EMAIL SPAM DETECTION - XGBoost Edition")
    print("Using only 3 features: Character count, Word count, Sentence count")
    print("="*65)
    
    # Initialize and train
    classifier = MinimalEmailSpamXGBoost()
    model = classifier.train()
    
    print(f"\n✅ Minimal XGBoost model ready!")
    print(f"📊 Features used: Only 3 basic length metrics")
    print(f"📁 Results visualization saved")
    print(f"🔍 Feature importance analysis included")
    
    return classifier

def compare_models_side_by_side():
    """Optional: Compare SVM vs XGBoost performance"""
    print("\n🔄 OPTIONAL: Model Comparison")
    print("="*40)
    print("This XGBoost version provides:")
    print("• Feature importance rankings")
    print("• Probability estimates for predictions")
    print("• Generally better performance on tabular data")
    print("• Built-in regularization")
    print("• Handles missing values automatically")
    
    print("\nCompared to SVM version:")
    print("• XGBoost is typically faster on large datasets")
    print("• Provides more interpretable results")
    print("• Less sensitive to feature scaling")
    print("• Better handles non-linear relationships")

if __name__ == "__main__":
    # Run the minimal XGBoost version
    spam_detector = run_minimal_spam_detection()
    
    # Show feature importance
    importance = spam_detector.get_feature_importance()
    
    # Optional: Test with sample emails
    print(f"\n🧪 Test with sample emails? (y/n): ", end="")
    try:
        test_choice = input().strip().lower()
        if test_choice in ['y', 'yes']:
            
            # Test with a short spam-like text
            print(f"\n--- Testing Sample Emails ---")
            
            spam_sample = "WIN $1000 NOW! Click here!"
            print(f"\nSample 1 (likely spam): '{spam_sample}'")
            spam_detector.predict_single(spam_sample)
            
            ham_sample = "Hi John, I hope you're doing well. I wanted to follow up on our meeting last week regarding the project timeline. Could we schedule another meeting to discuss the next steps? Let me know what works for your schedule. Best regards, Sarah"
            print(f"\nSample 2 (likely ham): '{ham_sample[:60]}...'")
            spam_detector.predict_single(ham_sample)
            
    except:
        pass
    
    # Show comparison info
    compare_models_side_by_side()
    
    print(f"\n👋 Done with XGBoost version!")