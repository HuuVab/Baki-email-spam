import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import seaborn as sns
import re

class MinimalEmailSpamLightGBM:
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
        """Train the minimal LightGBM model"""
        print("⚡ Training minimal LightGBM...")
        
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
        
        # Create and train LightGBM model
        # Note: LightGBM typically doesn't need feature scaling, but we'll keep it optional
        self.model = Pipeline([
            ('scaler', StandardScaler()),  # Optional for LightGBM
            ('lgb', lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,          # LightGBM specific parameter
                subsample=0.8,          # Row sampling
                colsample_bytree=0.8,   # Column sampling
                random_state=42,
                verbosity=-1,           # Reduce output noise
                objective='binary',     # Binary classification
                metric='binary_logloss'
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
        
        # Get feature importance from LightGBM
        lgb_model = self.model.named_steps['lgb']
        feature_importance = lgb_model.feature_importances_
        
        print(f"\n🔍 FEATURE IMPORTANCE (LightGBM):")
        print("="*42)
        for name, importance in zip(self.feature_names, feature_importance):
            print(f"{name:<15}: {importance:.0f} ({importance/sum(feature_importance)*100:.2f}%)")
        
        # Print LightGBM specific info
        print(f"\n⚡ LIGHTGBM MODEL INFO:")
        print("="*30)
        print(f"Number of Trees: {lgb_model.n_estimators}")
        print(f"Max Depth: {lgb_model.max_depth}")
        print(f"Number of Leaves: {lgb_model.num_leaves}")
        print(f"Learning Rate: {lgb_model.learning_rate}")
        print(f"Subsample Ratio: {lgb_model.subsample}")
        print(f"Feature Fraction: {lgb_model.colsample_bytree}")
        
        # Create visualizations
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax1)
        ax1.set_title(f'Confusion Matrix (LightGBM)\nAccuracy: {accuracy:.4f}')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Feature Importance
        ax2 = fig.add_subplot(gs[0, 1])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for LightGBM
        bars = ax2.bar(self.feature_names, feature_importance, color=colors, alpha=0.8)
        ax2.set_title('LightGBM Feature Importance')
        ax2.set_ylabel('Importance Score')
        ax2.set_xlabel('Features')
        
        # Add value labels on bars
        for bar, importance in zip(bars, feature_importance):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(feature_importance)*0.01,
                    f'{importance:.0f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Feature Importance Pie Chart (LightGBM specific)
        ax3 = fig.add_subplot(gs[1, 0])
        wedges, texts, autotexts = ax3.pie(feature_importance, labels=self.feature_names, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Feature Importance Distribution')
        
        # 4. Feature Distribution by Class
        # Combine train and test for full picture
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([pd.Series(y_train), pd.Series(y_test)])
        
        ham_data = X_full[y_full == 0]
        spam_data = X_full[y_full == 1]
        
        # Box plot comparison
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Prepare data for box plot
        char_data = [ham_data['char_count'].values, spam_data['char_count'].values]
        box_plot = ax4.boxplot(char_data, labels=['Ham', 'Spam'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        ax4.set_title('Character Count Distribution')
        ax4.set_ylabel('Character Count')
        ax4.set_ylim(0, 3000)  # Limit for better visualization
        
        # 5. Training Progress Simulation (Feature Importance over "iterations")
        ax5 = fig.add_subplot(gs[2, 0])
        
        # Simulate how feature importance might evolve (for visualization)
        iterations = np.arange(1, 11)
        importance_evolution = np.array([
            feature_importance * (0.5 + 0.05 * i) for i in iterations
        ]).T
        
        for i, (feature, color) in enumerate(zip(self.feature_names, colors)):
            ax5.plot(iterations, importance_evolution[i], marker='o', 
                    label=feature, color=color, linewidth=2)
        
        ax5.set_title('Feature Importance Evolution (Simulated)')
        ax5.set_xlabel('Iteration (x10)')
        ax5.set_ylabel('Importance Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Average feature comparison with LightGBM styling
        ax6 = fig.add_subplot(gs[2, 1])
        ham_avg = ham_data.mean()
        spam_avg = spam_data.mean()
        
        features = ['Characters', 'Words', 'Sentences']
        ham_values = [ham_avg['char_count'], ham_avg['word_count'], ham_avg['sentence_count']]
        spam_values = [spam_avg['char_count'], spam_avg['word_count'], spam_avg['sentence_count']]
        
        x = np.arange(len(features))
        width = 0.35
        
        ax6.bar(x - width/2, ham_values, width, label='Ham', color='#2ca02c', alpha=0.8)
        ax6.bar(x + width/2, spam_values, width, label='Spam', color='#d62728', alpha=0.8)
        
        ax6.set_title('Average Feature Values')
        ax6.set_xlabel('Features')
        ax6.set_ylabel('Average Count')
        ax6.set_xticks(x)
        ax6.set_xticklabels(features)
        ax6.legend()
        
        # Add value labels on bars
        for i, (ham_val, spam_val) in enumerate(zip(ham_values, spam_values)):
            ax6.text(i - width/2, ham_val + max(ham_values)*0.01, f'{ham_val:.0f}', 
                    ha='center', va='bottom', fontsize=9)
            ax6.text(i + width/2, spam_val + max(spam_values)*0.01, f'{spam_val:.0f}', 
                    ha='center', va='bottom', fontsize=9)
        
        # Save plot
        output_path = "E:\\Baki\\minimal_lightgbm_results.png"
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
        
        print(f"\n📧 Email Analysis (LightGBM):")
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
        
        lgb_model = self.model.named_steps['lgb']
        importance_dict = dict(zip(self.feature_names, lgb_model.feature_importances_))
        
        return importance_dict
    
    def get_model_info(self):
        """Get detailed LightGBM model information"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        lgb_model = self.model.named_steps['lgb']
        
        model_info = {
            'n_estimators': lgb_model.n_estimators,
            'max_depth': lgb_model.max_depth,
            'num_leaves': lgb_model.num_leaves,
            'learning_rate': lgb_model.learning_rate,
            'subsample': lgb_model.subsample,
            'colsample_bytree': lgb_model.colsample_bytree,
            'objective': lgb_model.objective,
            'metric': lgb_model.metric
        }
        
        return model_info
    
    def plot_feature_importance_detailed(self):
        """Create a detailed feature importance plot"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        lgb_model = self.model.named_steps['lgb']
        
        # Plot using LightGBM's built-in plotting function
        plt.figure(figsize=(10, 6))
        lgb.plot_importance(lgb_model, importance_type='split', max_num_features=3)
        plt.title('LightGBM Feature Importance (Split-based)')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        lgb.plot_importance(lgb_model, importance_type='gain', max_num_features=3)
        plt.title('LightGBM Feature Importance (Gain-based)')
        plt.tight_layout()
        plt.show()

def run_minimal_spam_detection():
    """Run the minimal spam detection with LightGBM"""
    print("⚡ MINIMAL EMAIL SPAM DETECTION - LightGBM Edition")
    print("Using only 3 features: Character count, Word count, Sentence count")
    print("="*70)
    
    # Initialize and train
    classifier = MinimalEmailSpamLightGBM()
    model = classifier.train()
    
    print(f"\n✅ Minimal LightGBM model ready!")
    print(f"📊 Features used: Only 3 basic length metrics")
    print(f"📁 Results visualization saved")
    print(f"🔍 Feature importance analysis included")
    print(f"⚡ LightGBM optimizations applied")
    
    return classifier

def compare_lightgbm_advantages():
    """Display advantages of LightGBM approach"""
    print("\n⚡ LIGHTGBM ADVANTAGES:")
    print("="*35)
    print("🚀 Speed & Efficiency:")
    print("   • Faster training than XGBoost")
    print("   • Lower memory consumption")
    print("   • GPU acceleration support")
    print("   • Efficient handling of large datasets")
    
    print("\n🎯 Performance:")
    print("   • Often achieves better accuracy")
    print("   • Built-in regularization")
    print("   • Handles overfitting well")
    print("   • Good with small datasets")
    
    print("\n🔧 Features:")
    print("   • Leaf-wise tree growth (vs level-wise)")
    print("   • Advanced feature importance metrics")
    print("   • Automatic handling of categorical features")
    print("   • Network communication for distributed training")
    
    print("\n📊 Interpretability:")
    print("   • Multiple importance metrics (split, gain)")
    print("   • Feature interaction detection")
    print("   • Built-in plotting functions")
    print("   • SHAP value support")
    
    print("\n⚠️  Considerations:")
    print("   • Can overfit on small datasets")
    print("   • Sensitive to hyperparameters")
    print("   • May need parameter tuning for optimal results")

def benchmark_comparison():
    """Compare different model characteristics"""
    print("\n📊 MODEL COMPARISON SUMMARY:")
    print("="*40)
    print(f"{'Model':<12} {'Speed':<8} {'Accuracy':<10} {'Interpretability':<15}")
    print("-" * 50)
    print(f"{'SVM':<12} {'Medium':<8} {'Good':<10} {'Low':<15}")
    print(f"{'XGBoost':<12} {'Medium':<8} {'Very Good':<10} {'Medium':<15}")
    print(f"{'DecisionTree':<12} {'Fast':<8} {'Good':<10} {'Very High':<15}")
    print(f"{'LightGBM':<12} {'Very Fast':<8} {'Very Good':<10} {'High':<15}")

if __name__ == "__main__":
    # Run the minimal LightGBM version
    spam_detector = run_minimal_spam_detection()
    
    # Show feature importance
    importance = spam_detector.get_feature_importance()
    
    # Show model info
    model_info = spam_detector.get_model_info()
    print(f"\n⚡ LightGBM Model Configuration:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Optional: Show detailed feature importance plots
    print(f"\n📊 Show detailed feature importance plots? (y/n): ", end="")
    try:
        plot_choice = input().strip().lower()
        if plot_choice in ['y', 'yes']:
            spam_detector.plot_feature_importance_detailed()
    except:
        pass
    
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
    
    # Show LightGBM advantages and comparison
    compare_lightgbm_advantages()
    benchmark_comparison()
    
    print(f"\n👋 Done with LightGBM version!")