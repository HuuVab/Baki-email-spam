import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import seaborn as sns
import re

class MinimalEmailSpamDecisionTree:
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
        """Train the minimal Decision Tree model"""
        print("🌳 Training minimal Decision Tree...")
        
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
        
        # Create and train Decision Tree model
        # Note: Decision Trees don't usually need scaling, but we'll keep it optional
        self.model = Pipeline([
            ('scaler', StandardScaler()),  # Optional for Decision Trees
            ('dt', DecisionTreeClassifier(
                max_depth=5,           # Limit depth to prevent overfitting
                min_samples_split=20,  # Minimum samples to split a node
                min_samples_leaf=10,   # Minimum samples in a leaf
                random_state=42,
                criterion='gini'       # Could also use 'entropy'
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
        
        # Get feature importance from Decision Tree
        dt_model = self.model.named_steps['dt']
        feature_importance = dt_model.feature_importances_
        
        print(f"\n🔍 FEATURE IMPORTANCE (Decision Tree):")
        print("="*45)
        for name, importance in zip(self.feature_names, feature_importance):
            print(f"{name:<15}: {importance:.4f} ({importance*100:.2f}%)")
        
        # Print tree depth and structure info
        print(f"\n🌳 TREE STRUCTURE INFO:")
        print("="*30)
        print(f"Tree Depth: {dt_model.get_depth()}")
        print(f"Number of Leaves: {dt_model.get_n_leaves()}")
        print(f"Number of Nodes: {dt_model.tree_.node_count}")
        
        # Create visualizations
        fig = plt.figure(figsize=(16, 12))
        
        # Create a 2x3 grid
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax1)
        ax1.set_title(f'Confusion Matrix (Decision Tree)\nAccuracy: {accuracy:.4f}')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Feature Importance
        ax2 = fig.add_subplot(gs[0, 1])
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        bars = ax2.bar(self.feature_names, feature_importance, color=colors, alpha=0.7)
        ax2.set_title('Decision Tree Feature Importance')
        ax2.set_ylabel('Importance Score')
        ax2.set_xlabel('Features')
        
        # Add value labels on bars
        for bar, importance in zip(bars, feature_importance):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Feature Distribution by Class
        # Combine train and test for full picture
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([pd.Series(y_train), pd.Series(y_test)])
        
        ham_data = X_full[y_full == 0]
        spam_data = X_full[y_full == 1]
        
        # Character count distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(ham_data['char_count'], bins=50, alpha=0.7, label='Ham', color='green', density=True)
        ax3.hist(spam_data['char_count'], bins=50, alpha=0.7, label='Spam', color='red', density=True)
        ax3.set_xlim(0, 5000)
        ax3.set_title('Character Count Distribution')
        ax3.set_xlabel('Characters')
        ax3.set_ylabel('Density')
        ax3.legend()
        
        # 4. Average feature comparison
        ax4 = fig.add_subplot(gs[1, 1])
        ham_avg = ham_data.mean()
        spam_avg = spam_data.mean()
        
        features = ['Characters', 'Words', 'Sentences']
        ham_values = [ham_avg['char_count'], ham_avg['word_count'], ham_avg['sentence_count']]
        spam_values = [spam_avg['char_count'], spam_avg['word_count'], spam_avg['sentence_count']]
        
        x = np.arange(len(features))
        width = 0.35
        
        ax4.bar(x - width/2, ham_values, width, label='Ham', color='green', alpha=0.7)
        ax4.bar(x + width/2, spam_values, width, label='Spam', color='red', alpha=0.7)
        
        ax4.set_title('Average Feature Values')
        ax4.set_xlabel('Features')
        ax4.set_ylabel('Average Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels(features)
        ax4.legend()
        
        # Add value labels on bars
        for i, (ham_val, spam_val) in enumerate(zip(ham_values, spam_values)):
            ax4.text(i - width/2, ham_val + max(ham_values)*0.01, f'{ham_val:.0f}', 
                    ha='center', va='bottom', fontsize=9)
            ax4.text(i + width/2, spam_val + max(spam_values)*0.01, f'{spam_val:.0f}', 
                    ha='center', va='bottom', fontsize=9)
        
        # 5. Decision Tree Visualization (simplified)
        ax5 = fig.add_subplot(gs[2, :])
        
        # Plot a simplified version of the tree (max_depth=3 for visualization)
        plot_tree(dt_model, max_depth=3, 
                 feature_names=self.feature_names,
                 class_names=['Ham', 'Spam'],
                 filled=True, rounded=True, fontsize=10, ax=ax5)
        ax5.set_title('Decision Tree Structure (Simplified View - Max Depth 3)')
        
        # Save plot
        output_path = "E:\\Baki\\minimal_decision_tree_results.png"
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
        
        print(f"\n📧 Email Analysis (Decision Tree):")
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
        
        dt_model = self.model.named_steps['dt']
        importance_dict = dict(zip(self.feature_names, dt_model.feature_importances_))
        
        return importance_dict
    
    def get_decision_rules(self, max_depth=3):
        """Extract and display decision rules from the tree"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        dt_model = self.model.named_steps['dt']
        
        print(f"\n🌳 DECISION RULES (Max Depth {max_depth}):")
        print("="*50)
        
        # Get text representation of the tree
        tree_rules = export_text(dt_model, 
                                feature_names=self.feature_names,
                                class_names=['Ham', 'Spam'],
                                max_depth=max_depth)
        print(tree_rules)
        
        return tree_rules
    
    def analyze_tree_structure(self):
        """Provide detailed analysis of the tree structure"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        dt_model = self.model.named_steps['dt']
        
        print(f"\n🔍 DETAILED TREE ANALYSIS:")
        print("="*35)
        print(f"Max Depth Used: {dt_model.get_depth()}")
        print(f"Total Nodes: {dt_model.tree_.node_count}")
        print(f"Total Leaves: {dt_model.get_n_leaves()}")
        print(f"Tree Parameters:")
        print(f"  - Max Depth Setting: {dt_model.max_depth}")
        print(f"  - Min Samples Split: {dt_model.min_samples_split}")
        print(f"  - Min Samples Leaf: {dt_model.min_samples_leaf}")
        print(f"  - Criterion: {dt_model.criterion}")

def run_minimal_spam_detection():
    """Run the minimal spam detection with Decision Tree"""
    print("🌳 MINIMAL EMAIL SPAM DETECTION - Decision Tree Edition")
    print("Using only 3 features: Character count, Word count, Sentence count")
    print("="*70)
    
    # Initialize and train
    classifier = MinimalEmailSpamDecisionTree()
    model = classifier.train()
    
    print(f"\n✅ Minimal Decision Tree model ready!")
    print(f"📊 Features used: Only 3 basic length metrics")
    print(f"📁 Results visualization saved")
    print(f"🔍 Feature importance analysis included")
    print(f"🌳 Tree structure visualization included")
    
    return classifier

def compare_decision_tree_advantages():
    """Display advantages of Decision Tree approach"""
    print("\n🎯 DECISION TREE ADVANTAGES:")
    print("="*40)
    print("✅ Highly Interpretable:")
    print("   • Can see exact decision rules")
    print("   • Easy to explain to non-technical users")
    print("   • Visual tree structure shows logic flow")
    
    print("\n✅ No Assumptions About Data:")
    print("   • Works with non-linear relationships")
    print("   • No need for feature scaling")
    print("   • Handles mixed data types well")
    
    print("\n✅ Built-in Feature Selection:")
    print("   • Automatically identifies important features")
    print("   • Ignores irrelevant features")
    print("   • Shows feature importance scores")
    
    print("\n✅ Fast and Efficient:")
    print("   • Quick training and prediction")
    print("   • Low computational requirements")
    print("   • Good for real-time applications")
    
    print("\n⚠️  Considerations:")
    print("   • Can overfit with deep trees")
    print("   • May be less stable than ensemble methods")
    print("   • Performance can vary with small data changes")

if __name__ == "__main__":
    # Run the minimal Decision Tree version
    spam_detector = run_minimal_spam_detection()
    
    # Show feature importance
    importance = spam_detector.get_feature_importance()
    
    # Show tree structure analysis
    spam_detector.analyze_tree_structure()
    
    # Show decision rules
    spam_detector.get_decision_rules(max_depth=3)
    
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
    
    # Show Decision Tree advantages
    compare_decision_tree_advantages()
    
    print(f"\n👋 Done with Decision Tree version!")