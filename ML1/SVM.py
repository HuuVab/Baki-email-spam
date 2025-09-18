import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def build_minimal_svm_classifier(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """
    Minimal SVM spam classifier with essential saving functionality
    """
    print("🚀 Minimal SVM Spam Classifier")
    print("="*40)
    
    # Load data
    print("📁 Loading features...")
    df = pd.read_csv(csv_file)
    feature_columns = [col for col in df.columns if col != 'label']
    
    X = df[feature_columns]
    y = df['label']
    
    # Fill missing values if any
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data: {len(X_train)} train, {len(X_test)} test samples")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('feature_selection', SelectKBest(f_classif, k=15)),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    # Minimal parameter grid for faster training
    param_grid = {
        'feature_selection__k': [10, 15, 20],
        'svm__C': [1, 10, 100],
        'svm__kernel': ['linear', 'rbf'],
        'svm__class_weight': ['balanced']
    }
    
    # Grid search
    print("🔍 Training model...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_params': grid_search.best_params_,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Get important features
    feature_selector = best_model.named_steps['feature_selection']
    selected_mask = feature_selector.get_support()
    selected_features = [feature_columns[i] for i, selected in enumerate(selected_mask) if selected]
    
    # Feature importance scores
    feature_scores = feature_selector.scores_
    important_features = {
        'selected_features': selected_features,
        'feature_scores': {feature_columns[i]: float(feature_scores[i]) 
                          for i in range(len(feature_columns))},
        'top_10_features': sorted(
            [(feature_columns[i], float(feature_scores[i])) 
             for i in range(len(feature_columns))],
            key=lambda x: x[1], reverse=True
        )[:10]
    }
    
    # Print results
    print(f"\n📊 Results:")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1_score']:.4f}")
    print(f"   ROC AUC:   {results['roc_auc']:.4f}")
    
    print(f"\n🎯 Top 5 Important Features:")
    for i, (feature, score) in enumerate(important_features['top_10_features'][:5], 1):
        print(f"   {i}. {feature}: {score:.2f}")
    
    # Save everything
    save_results(best_model, results, important_features, y_test, y_pred)
    
    return best_model, results, important_features

def save_results(model, results, features, y_test, y_pred):
    """
    Save model and results to files
    """
    base_path = "E:\\Baki\\"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save the trained model
    model_path = f"{base_path}svm_model_{timestamp}.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # 2. Save test results as JSON
    results_path = f"{base_path}test_results_{timestamp}.json"
    
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
    
    # 3. Save important features as JSON
    features_path = f"{base_path}important_features_{timestamp}.json"
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"💾 Features saved: {features_path}")
    
    # 4. Create simple performance plot
    create_simple_plot(y_test, y_pred, results, features, timestamp)
    
    # 5. Save summary text file
    save_summary_text(results, features, timestamp)

def create_simple_plot(y_test, y_pred, results, features, timestamp):
    """
    Create a simple performance visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ax1 = axes[0]
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title('Confusion Matrix')
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
    
    plt.tight_layout()
    plot_path = f"E:\\Baki\\svm_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved: {plot_path}")
    plt.show()

def save_summary_text(results, features, timestamp):
    """
    Save a readable summary text file
    """
    summary_path = f"E:\\Baki\\model_summary_{timestamp}.txt"
    
    with open(summary_path, 'w') as f:
        f.write("SVM SPAM CLASSIFIER - MODEL SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Training Date: {results['timestamp']}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*25 + "\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)\n")
        f.write(f"Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)\n")
        f.write(f"F1-Score:  {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)\n")
        f.write(f"ROC AUC:   {results['roc_auc']:.4f} ({results['roc_auc']*100:.2f}%)\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-"*20 + "\n")
        cm = results['confusion_matrix']
        f.write(f"True Negatives (Ham→Ham):   {cm['true_negatives']}\n")
        f.write(f"False Positives (Ham→Spam): {cm['false_positives']}\n")
        f.write(f"False Negatives (Spam→Ham): {cm['false_negatives']}\n")
        f.write(f"True Positives (Spam→Spam): {cm['true_positives']}\n\n")
        
        f.write("BEST MODEL PARAMETERS:\n")
        f.write("-"*25 + "\n")
        for param, value in results['best_params'].items():
            f.write(f"{param}: {value}\n")
        
        f.write(f"\nTOP 10 IMPORTANT FEATURES:\n")
        f.write("-"*30 + "\n")
        for i, (feature, score) in enumerate(features['top_10_features'], 1):
            f.write(f"{i:2d}. {feature:<25} | Score: {score:.2f}\n")
    
    print(f"📄 Summary saved: {summary_path}")

# Main execution
if __name__ == "__main__":
    print("🚀 MINIMAL SVM SPAM CLASSIFIER")
    print("="*40)
    
    try:
        model, results, features = build_minimal_svm_classifier()
        
        print(f"\n✅ TRAINING COMPLETE!")
        print("="*30)
        print(f"📊 F1-Score: {results['f1_score']:.1%}")
        print(f"🎯 Accuracy: {results['accuracy']:.1%}")
        print(f"💾 All files saved with timestamp")
        print(f"🎉 Ready for deployment!")
        
    except FileNotFoundError:
        print("❌ Error: ml_features_matrix.csv not found!")
        print("💡 Please check the file path")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    input("\nPress Enter to exit...")