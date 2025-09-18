import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def build_minimal_nb_classifier(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """
    Minimal Naive Bayes spam classifier with essential saving functionality
    """
    print("🧠 Minimal Naive Bayes Spam Classifier")
    print("="*42)
    
    # Load data
    print("📁 Loading features...")
    df = pd.read_csv(csv_file)
    feature_columns = [col for col in df.columns if col != 'label']
    
    X = df[feature_columns]
    y = df['label']
    
    # Handle missing values (fill with 0 for NB)
    if X.isnull().sum().sum() > 0:
        X = X.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data: {len(X_train)} train, {len(X_test)} test samples")
    
    # Test all Naive Bayes variants
    models = {
        'Gaussian NB': GaussianNB(),
        'Multinomial NB': MultinomialNB(),
        'Bernoulli NB': BernoulliNB()
    }
    
    print("🔍 Testing NB variants...")
    
    results = {}
    for name, model in models.items():
        # Handle negative values for Multinomial NB
        if name == 'Multinomial NB':
            # Make features non-negative for Multinomial NB
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
        f1 = f1_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1,
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"   {name:<15} | F1: {f1:.4f}")
    
    # Select best model based on F1 score
    best_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_model = results[best_name]['model']
    best_results = results[best_name]
    
    print(f"\n🏆 Best model: {best_name}")
    print(f"   F1-Score: {best_results['f1_score']:.4f}")
    
    # Cross-validation check
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='f1')
    
    # Prepare final results
    final_results = {
        'best_variant': best_name,
        'accuracy': best_results['accuracy'],
        'precision': best_results['precision'],
        'recall': best_results['recall'],
        'f1_score': best_results['f1_score'],
        'roc_auc': best_results['roc_auc'],
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'all_variants': {name: {'f1': res['f1_score'], 'accuracy': res['accuracy']} 
                        for name, res in results.items()},
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Get feature insights (basic for NB)
    feature_insights = get_nb_feature_insights(best_model, feature_columns, X_train, y_train, best_name)
    
    # Print results
    print(f"\n📊 Results:")
    print(f"   Accuracy:  {final_results['accuracy']:.4f}")
    print(f"   Precision: {final_results['precision']:.4f}")
    print(f"   Recall:    {final_results['recall']:.4f}")
    print(f"   F1-Score:  {final_results['f1_score']:.4f}")
    print(f"   ROC AUC:   {final_results['roc_auc']:.4f}")
    print(f"   CV F1:     {final_results['cv_score']:.4f} (+/- {final_results['cv_std']:.4f})")
    
    # Save everything
    save_nb_results(best_model, final_results, feature_insights, y_test, best_results['y_pred'])
    
    return best_model, final_results, feature_insights

def get_nb_feature_insights(model, feature_columns, X_train, y_train, model_name):
    """
    Extract basic insights from Naive Bayes model
    """
    insights = {
        'model_type': model_name,
        'n_features': len(feature_columns),
        'feature_names': feature_columns
    }
    
    # For Gaussian NB, we can get feature statistics
    if model_name == 'Gaussian NB':
        # Get class-wise feature means and variances
        class_means = {}
        class_vars = {}
        
        for class_idx, class_label in enumerate([0, 1]):  # Ham, Spam
            class_mask = y_train == class_label
            class_data = X_train[class_mask]
            
            means = class_data.mean()
            variances = class_data.var()
            
            class_means[f'class_{class_label}'] = means.to_dict()
            class_vars[f'class_{class_label}'] = variances.to_dict()
        
        # Find features with biggest difference between classes
        feature_diffs = []
        for feature in feature_columns:
            diff = abs(class_means['class_1'][feature] - class_means['class_0'][feature])
            feature_diffs.append((feature, diff))
        
        feature_diffs.sort(key=lambda x: x[1], reverse=True)
        
        insights['class_means'] = class_means
        insights['class_variances'] = class_vars
        insights['top_discriminative_features'] = feature_diffs[:10]
    
    # For Bernoulli NB, we can get feature log probabilities
    elif model_name == 'Bernoulli NB':
        # Get feature log probabilities for each class
        feature_log_probs = {}
        for class_idx, class_label in enumerate(['Ham', 'Spam']):
            feature_log_probs[class_label] = model.feature_log_prob_[class_idx].tolist()
        
        insights['feature_log_probabilities'] = feature_log_probs
    
    # For Multinomial NB
    elif model_name == 'Multinomial NB':
        # Get feature log probabilities
        feature_log_probs = {}
        for class_idx, class_label in enumerate(['Ham', 'Spam']):
            feature_log_probs[class_label] = model.feature_log_prob_[class_idx].tolist()
        
        insights['feature_log_probabilities'] = feature_log_probs
    
    return insights

def save_nb_results(model, results, insights, y_test, y_pred):
    """
    Save Naive Bayes model and results to files
    """
    base_path = "E:\\Baki\\"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save the trained model
    model_path = f"{base_path}nb_model_{timestamp}.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # 2. Save test results as JSON
    results_path = f"{base_path}nb_results_{timestamp}.json"
    
    # Add confusion matrix to results
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = {
        'true_negatives': int(cm[0,0]),
        'false_positives': int(cm[0,1]),
        'false_negatives': int(cm[1,0]),
        'true_positives': int(cm[1,1])
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved: {results_path}")
    
    # 3. Save feature insights as JSON
    insights_path = f"{base_path}nb_insights_{timestamp}.json"
    with open(insights_path, 'w') as f:
        json.dump(insights, f, indent=2)
    print(f"💾 Insights saved: {insights_path}")
    
    # 4. Create simple performance plot
    create_nb_plot(y_test, y_pred, results, insights, timestamp)
    
    # 5. Save summary text file
    save_nb_summary(results, insights, timestamp)

def create_nb_plot(y_test, y_pred, results, insights, timestamp):
    """
    Create simple Naive Bayes performance visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ax1 = axes[0]
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title(f'Confusion Matrix\n({results["best_variant"]})')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Ham', 'Spam'])
    ax1.set_yticklabels(['Ham', 'Spam'])
    
    # Performance Metrics
    ax2 = axes[1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    values = [results['accuracy'], results['precision'], results['recall'], 
              results['f1_score'], results['roc_auc']]
    
    bars = ax2.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'gold'])
    ax2.set_title('Performance Metrics')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Variant Comparison
    ax3 = axes[2]
    variants = list(results['all_variants'].keys())
    f1_scores = [results['all_variants'][v]['f1'] for v in variants]
    
    # Highlight best variant
    colors = ['gold' if v == results['best_variant'] else 'lightblue' for v in variants]
    
    bars = ax3.bar(variants, f1_scores, color=colors, alpha=0.8)
    ax3.set_title('NB Variants Comparison')
    ax3.set_ylabel('F1 Score')
    ax3.set_ylim(0, max(f1_scores) * 1.1)
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plot_path = f"E:\\Baki\\nb_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved: {plot_path}")
    plt.show()

def save_nb_summary(results, insights, timestamp):
    """
    Save a readable Naive Bayes summary text file
    """
    summary_path = f"E:\\Baki\\nb_summary_{timestamp}.txt"
    
    with open(summary_path, 'w') as f:
        f.write("NAIVE BAYES SPAM CLASSIFIER - MODEL SUMMARY\n")
        f.write("="*55 + "\n\n")
        f.write(f"Training Date: {results['timestamp']}\n")
        f.write(f"Best Variant: {results['best_variant']}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*25 + "\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)\n")
        f.write(f"Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)\n")
        f.write(f"F1-Score:  {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)\n")
        f.write(f"ROC AUC:   {results['roc_auc']:.4f} ({results['roc_auc']*100:.2f}%)\n")
        f.write(f"CV F1:     {results['cv_score']:.4f} (+/- {results['cv_std']:.4f})\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-"*20 + "\n")
        cm = results['confusion_matrix']
        f.write(f"True Negatives (Ham→Ham):   {cm['true_negatives']}\n")
        f.write(f"False Positives (Ham→Spam): {cm['false_positives']}\n")
        f.write(f"False Negatives (Spam→Ham): {cm['false_negatives']}\n")
        f.write(f"True Positives (Spam→Spam): {cm['true_positives']}\n\n")
        
        f.write("VARIANT COMPARISON:\n")
        f.write("-"*20 + "\n")
        for variant, scores in results['all_variants'].items():
            marker = "🏆 " if variant == results['best_variant'] else "   "
            f.write(f"{marker}{variant:<15} | F1: {scores['f1']:.4f} | Acc: {scores['accuracy']:.4f}\n")
        
        f.write(f"\nMODEL CHARACTERISTICS:\n")
        f.write("-"*22 + "\n")
        f.write(f"Features Used: {insights['n_features']}\n")
        f.write(f"Model Type: {insights['model_type']}\n")
        
        # Add discriminative features if available
        if 'top_discriminative_features' in insights:
            f.write(f"\nTOP 10 DISCRIMINATIVE FEATURES:\n")
            f.write("-"*35 + "\n")
            for i, (feature, diff) in enumerate(insights['top_discriminative_features'], 1):
                f.write(f"{i:2d}. {feature:<25} | Difference: {diff:.4f}\n")
        
        f.write(f"\nNAIVE BAYES ADVANTAGES:\n")
        f.write("-"*25 + "\n")
        f.write("✓ Extremely fast training and prediction\n")
        f.write("✓ Works well with small datasets\n")
        f.write("✓ Simple and interpretable\n")
        f.write("✓ Good baseline performance\n")
        f.write("✓ Handles irrelevant features well\n")
        f.write("✓ No hyperparameter tuning needed\n")
        
        f.write(f"\nLIMITATIONS:\n")
        f.write("-"*12 + "\n")
        f.write("⚠ Assumes feature independence\n")
        f.write("⚠ Can be outperformed by ensemble methods\n")
        f.write("⚠ Sensitive to skewed data\n")
    
    print(f"📄 Summary saved: {summary_path}")

# Main execution
if __name__ == "__main__":
    print("🧠 MINIMAL NAIVE BAYES SPAM CLASSIFIER")
    print("="*45)
    
    try:
        model, results, insights = build_minimal_nb_classifier()
        
        print(f"\n✅ TRAINING COMPLETE!")
        print("="*30)
        print(f"📊 F1-Score: {results['f1_score']:.1%}")
        print(f"🎯 Accuracy: {results['accuracy']:.1%}")
        print(f"🧠 Best variant: {results['best_variant']}")
        print(f"⚡ CV F1: {results['cv_score']:.4f}")
        print(f"💾 All files saved with timestamp")
        
        # Quick advantage summary
        print(f"\n🧠 Naive Bayes Advantages:")
        print(f"   ✅ Fastest training time")
        print(f"   ✅ Simple and interpretable")
        print(f"   ✅ No hyperparameter tuning")
        print(f"   ✅ Good with small datasets")
        print(f"   ✅ Excellent baseline model")
        print(f"🚀 Perfect for real-time deployment!")
        
    except FileNotFoundError:
        print("❌ Error: ml_features_matrix.csv not found!")
        print("💡 Please check the file path")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    input("\nPress Enter to exit...")