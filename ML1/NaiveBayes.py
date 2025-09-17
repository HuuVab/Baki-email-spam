import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

def build_naive_bayes_classifier(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """
    Build Naive Bayes spam classifier - minimal and fast
    """
    print("🧠 NAIVE BAYES SPAM DETECTION MODEL")
    print("="*50)
    
    # Load data
    print("📁 Loading features...")
    df = pd.read_csv(csv_file)
    
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']
    
    print(f"📊 Dataset: {len(df):,} emails, {len(feature_columns)} features")
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Test different Naive Bayes variants
    models = {
        'Gaussian NB': GaussianNB(),
        'Multinomial NB': MultinomialNB(),
        'Bernoulli NB': BernoulliNB()
    }
    
    print(f"\n🔍 Testing Naive Bayes variants...")
    
    results = {}
    for name, model in models.items():
        # Handle negative values for Multinomial NB
        if name == 'Multinomial NB':
            X_train_pos = np.abs(X_train)
            X_test_pos = np.abs(X_test)
            model.fit(X_train_pos, y_train)
            y_pred = model.predict(X_test_pos)
            y_pred_proba = model.predict_proba(X_test_pos)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"   {name:<15} | F1: {f1:.4f} | Acc: {accuracy:.4f} | AUC: {auc:.4f}")
    
    # Select best model
    best_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_model = results[best_name]['model']
    best_results = results[best_name]
    
    print(f"\n🏆 BEST MODEL: {best_name}")
    print("="*40)
    
    # Detailed performance
    y_pred = best_results['y_pred']
    y_pred_proba = best_results['y_pred_proba']
    
    print(f"🎯 Performance Metrics:")
    print(f"   • Accuracy:  {best_results['accuracy']:.4f} ({best_results['accuracy']*100:.2f}%)")
    print(f"   • Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"   • Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"   • F1-Score:  {best_results['f1']:.4f}")
    print(f"   • ROC AUC:   {best_results['auc']:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
    print(f"\n🔄 Cross-Validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Classification report
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Ham', 'Spam'], digits=4))
    
    # Create visualizations
    create_nb_plots(y_test, y_pred, y_pred_proba, best_name, results)
    
    # Save model
    model_path = "E:\\Baki\\naive_bayes_spam_classifier.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    return best_model, results, feature_columns

def create_nb_plots(y_test, y_pred, y_pred_proba, best_name, all_results):
    """Create minimal but informative plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Naive Bayes Spam Detection - {best_name}', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    ax = axes[0, 0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax.set_title('Confusion Matrix', fontweight='bold')
    
    # Add accuracy
    accuracy = accuracy_score(y_test, y_pred)
    ax.text(1, -0.1, f'Accuracy: {accuracy:.3f}', ha='center', transform=ax.transAxes)
    
    # 2. ROC Curve
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    ax.plot(fpr, tpr, color='red', lw=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Performance Comparison
    ax = axes[0, 2]
    models = list(all_results.keys())
    f1_scores = [all_results[model]['f1'] for model in models]
    colors = ['gold' if model == best_name else 'lightblue' for model in models]
    
    bars = ax.bar(models, f1_scores, color=colors, alpha=0.8)
    ax.set_ylabel('F1 Score')
    ax.set_title('NB Variants Comparison', fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add values on bars
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # 4. Prediction Distribution
    ax = axes[1, 0]
    ham_probs = y_pred_proba[y_test == 0]
    spam_probs = y_pred_proba[y_test == 1]
    
    ax.hist(ham_probs, bins=30, alpha=0.7, label='Ham', color='green', density=True)
    ax.hist(spam_probs, bins=30, alpha=0.7, label='Spam', color='red', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel('Spam Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 5. Metrics Summary
    ax = axes[1, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    scores = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        auc
    ]
    
    bars = ax.bar(metrics, scores, color=['skyblue', 'lightgreen', 'orange', 'pink', 'gold'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics', fontweight='bold')
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Model Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    tn, fp, fn, tp = cm.ravel()
    summary_text = f"""
NAIVE BAYES SUMMARY

Best Model: {best_name}
Dataset: {len(y_test):,} test samples

PERFORMANCE:
- Accuracy:  {accuracy_score(y_test, y_pred):.4f}
- Precision: {precision_score(y_test, y_pred):.4f}
- Recall:    {recall_score(y_test, y_pred):.4f}
- F1-Score:  {f1_score(y_test, y_pred):.4f}
- AUC:       {auc:.4f}

CONFUSION MATRIX:
- True Positives:  {tp:,}
- True Negatives:  {tn:,}
- False Positives: {fp:,}
- False Negatives: {fn:,}

ADVANTAGES:
✓ Fast training
✓ Simple & interpretable
✓ Good baseline performance
✓ Handles multiple classes well
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("E:\\Baki\\naive_bayes_analysis.png", dpi=300, bbox_inches='tight')
    print("💾 Naive Bayes analysis saved: E:\\Baki\\naive_bayes_analysis.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    print("🧠 NAIVE BAYES SPAM CLASSIFIER (MINIMAL VERSION)")
    print("="*60)
    
    try:
        model, results, features = build_naive_bayes_classifier()
        
        # Quick insights
        print(f"\n💡 NAIVE BAYES INSIGHTS:")
        print("="*35)
        print("✅ Advantages:")
        print("   • Extremely fast training & prediction")
        print("   • Works well with small datasets")
        print("   • Simple and interpretable")
        print("   • Good baseline performance")
        print("   • Handles irrelevant features well")
        
        print("\n⚠️  Limitations:")
        print("   • Assumes feature independence")
        print("   • Can be outperformed by ensemble methods")
        print("   • Sensitive to skewed data")
        
        # Best variant info
        best_variant = max(results.keys(), key=lambda k: results[k]['f1'])
        print(f"\n🏆 Best variant: {best_variant}")
        print(f"🎯 F1-Score: {results[best_variant]['f1']:.4f}")
        print(f"⚡ Perfect for: Real-time classification, baseline models")
        
        print(f"\n✅ MODEL READY!")
        print(f"💾 Saved: naive_bayes_spam_classifier.pkl")
        print(f"📊 Analysis: naive_bayes_analysis.png")
        
    except FileNotFoundError:
        print("❌ Error: ml_features_matrix.csv not found!")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print(f"\n🎉 Press Enter to exit...")
    try:
        input()
    except:
        pass