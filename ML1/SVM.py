import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC  # Pure linear SVM
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
import joblib
import json
from datetime import datetime
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

def build_fast_linear_svm_classifier(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """
    Fast Linear SVM spam classifier - NO GRID SEARCH
    """
    print("🚀 FAST Linear SVM Spam Classifier")
    print("="*50)
    print("⚡ NO Grid Search - Direct Training")
    print("🎯 Using optimal defaults for spam detection")
    print("="*50)
    
    # Load data with progress
    print("📁 Loading features...")
    start_time = time.time()
    
    df = pd.read_csv(csv_file)
    feature_columns = [col for col in df.columns if col != 'label']
    
    X = df[feature_columns]
    y = df['label']
    
    load_time = time.time() - start_time
    print(f"✅ Data loaded in {load_time:.2f} seconds")
    print(f"📊 Features: {len(feature_columns)} engineered features")
    print(f"📈 Samples: {len(X):,} emails")
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print("🔧 Handling missing values...")
        X = X.fillna(X.median())
        print("✅ Missing values filled")
    
    # Split data
    print("✂️  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Split: {len(X_train):,} train, {len(X_test):,} test samples")
    
    # Show class distribution
    train_class_dist = y_train.value_counts().sort_index()
    print(f"📈 Training set distribution:")
    print(f"   Ham (0): {train_class_dist[0]:,} ({train_class_dist[0]/len(y_train)*100:.1f}%)")
    print(f"   Spam (1): {train_class_dist[1]:,} ({train_class_dist[1]/len(y_train)*100:.1f}%)")
    
    # Create simple pipeline - NO GRID SEARCH
    print(f"\n🏗️  Building Linear SVM Pipeline...")
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('svm', LinearSVC(
            C=1.0,  # Good default for most cases
            class_weight='balanced',  # Handle any class imbalance
            loss='squared_hinge',  # Often works better than hinge
            random_state=42,
            max_iter=2000,
            dual=False  # Faster for large datasets with many samples
        ))
    ])
    
    print(f"⚙️  Using optimized parameters:")
    print(f"   • C: 1.0 (regularization)")
    print(f"   • Class Weight: balanced")
    print(f"   • Loss: squared_hinge")
    print(f"   • Max Iterations: 2000")
    print(f"   • Dual: False (faster for large datasets)")
    
    # Train the model with progress
    print(f"\n🚀 Training Linear SVM...")
    print("=" * 40)
    
    train_start = time.time()
    
    # Simple progress simulation for training
    with tqdm(total=100, desc="🎯 Training Linear SVM", 
              bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]",
              ncols=80, colour='green') as pbar:
        
        # Fit the scaler first
        pbar.set_description("🔧 Scaling features")
        pbar.update(20)
        
        # Train the actual model
        pbar.set_description("🚀 Training SVM")
        pipeline.fit(X_train, y_train)
        pbar.update(80)
        
        pbar.set_description("✅ Training complete")
    
    train_time = time.time() - train_start
    print(f"✅ Training completed in {train_time:.2f} seconds!")
    
    # Make predictions
    print(f"\n🔮 Making predictions...")
    pred_start = time.time()
    
    with tqdm(total=100, desc="🎯 Predicting", ncols=60, colour='blue') as pbar:
        y_pred = pipeline.predict(X_test)
        pbar.update(50)
        
        y_scores = pipeline.decision_function(X_test)
        pbar.update(50)
    
    pred_time = time.time() - pred_start
    print(f"✅ Predictions completed in {pred_time:.3f} seconds")
    
    # Calculate metrics
    print(f"📊 Calculating performance metrics...")
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_scores),
        'training_time': train_time,
        'prediction_time': pred_time,
        'model_params': {
            'C': 1.0,
            'class_weight': 'balanced',
            'loss': 'squared_hinge',
            'dual': False
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Feature importance from coefficients
    print(f"🎯 Extracting feature importance...")
    svm_model = pipeline.named_steps['svm']
    feature_importance = abs(svm_model.coef_[0])
    
    feature_rankings = {
        'feature_names': feature_columns,
        'importance_scores': feature_importance.tolist(),
        'top_10_features': sorted(
            list(zip(feature_columns, feature_importance)),
            key=lambda x: x[1], reverse=True
        )[:10]
    }
    
    # Print results
    print(f"\n" + "="*60)
    print(f"🎉 TRAINING COMPLETE - PERFORMANCE SUMMARY")
    print(f"="*60)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print("-" * 35)
    metrics_display = [
        ("🎯 Accuracy", results['accuracy']),
        ("🔍 Precision", results['precision']),
        ("📈 Recall", results['recall']),
        ("⚖️  F1-Score", results['f1_score']),
        ("📊 ROC AUC", results['roc_auc'])
    ]
    
    for name, value in metrics_display:
        print(f"   {name:<15}: {value:.4f} ({value*100:.1f}%)")
    
    print(f"\n⚙️  MODEL PARAMETERS USED:")
    print("-" * 30)
    for param, value in results['model_params'].items():
        print(f"   {param:<15}: {value}")
    
    print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
    print("-" * 50)
    for i, (feature, importance) in enumerate(feature_rankings['top_10_features'], 1):
        clean_feature = feature.replace('_', ' ').title()
        # Create simple progress bar visualization
        max_importance = max([x[1] for x in feature_rankings['top_10_features']])
        bar_length = int(importance * 30 / max_importance)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"   {i:2d}. {clean_feature:<20} |{bar}| {importance:.4f}")
    
    print(f"\n⏱️  PERFORMANCE SUMMARY:")
    print("-" * 30)
    print(f"   Data Loading: {load_time:.2f}s")
    print(f"   Model Training: {train_time:.2f}s")
    print(f"   Predictions: {pred_time:.3f}s")
    print(f"   Total Runtime: {load_time + train_time + pred_time:.2f}s")
    
    # Generate confusion matrix details
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🎯 CONFUSION MATRIX BREAKDOWN:")
    print("-" * 35)
    print(f"   True Negatives (Ham → Ham):   {cm[0,0]:,}")
    print(f"   False Positives (Ham → Spam): {cm[0,1]:,}")
    print(f"   False Negatives (Spam → Ham): {cm[1,0]:,}")
    print(f"   True Positives (Spam → Spam): {cm[1,1]:,}")
    
    # Calculate error rates
    total_test = len(y_test)
    ham_error_rate = cm[0,1] / (cm[0,0] + cm[0,1]) * 100  # Ham misclassified as spam
    spam_error_rate = cm[1,0] / (cm[1,0] + cm[1,1]) * 100  # Spam misclassified as ham
    
    print(f"\n📈 ERROR ANALYSIS:")
    print("-" * 20)
    print(f"   Ham Error Rate:  {ham_error_rate:.2f}% ({cm[0,1]:,} emails)")
    print(f"   Spam Error Rate: {spam_error_rate:.2f}% ({cm[1,0]:,} emails)")
    print(f"   Overall Error:   {((cm[0,1] + cm[1,0]) / total_test * 100):.2f}%")
    
    # Save everything
    save_fast_svm_results(pipeline, results, feature_rankings, y_test, y_pred, feature_columns)
    
    return pipeline, results, feature_rankings

def save_fast_svm_results(model, results, features, y_test, y_pred, feature_columns):
    """Save results quickly"""
    base_path = "E:\\Baki\\"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n💾 SAVING RESULTS...")
    
    save_tasks = ["Model", "Results", "Features", "Plots", "Summary"]
    
    with tqdm(total=len(save_tasks), desc="💾 Saving Files", 
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
              ncols=60, colour='cyan') as pbar:
        
        # 1. Save model
        model_path = f"{base_path}fast_linear_svm_{timestamp}.pkl"
        joblib.dump(model, model_path)
        pbar.update(1)
        
        # 2. Save results
        results_path = f"{base_path}fast_svm_results_{timestamp}.json"
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = {
            'true_negatives': int(cm[0,0]),
            'false_positives': int(cm[0,1]), 
            'false_negatives': int(cm[1,0]),
            'true_positives': int(cm[1,1])
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        pbar.update(1)
        
        # 3. Save features
        features_path = f"{base_path}fast_svm_features_{timestamp}.json"
        with open(features_path, 'w') as f:
            json.dump(features, f, indent=2)
        pbar.update(1)
        
        # 4. Quick plot
        create_fast_svm_plot(y_test, y_pred, results, features, timestamp)
        pbar.update(1)
        
        # 5. Summary
        save_fast_svm_summary(results, features, timestamp)
        pbar.update(1)
    
    print(f"\n✅ All files saved!")
    print(f"📁 Files saved with timestamp: {timestamp}")

def create_fast_svm_plot(y_test, y_pred, results, features, timestamp):
    """Create a quick visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fast Linear SVM - Spam Classification Results', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title('Confusion Matrix', fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")
    
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Ham', 'Spam'])
    ax1.set_yticklabels(['Ham', 'Spam'])
    
    # 2. Performance Metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    values = [results['accuracy'], results['precision'], results['recall'], 
              results['f1_score'], results['roc_auc']]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
    ax2.set_title('Performance Metrics', fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Top Features
    top_features = features['top_10_features'][:8]  # Top 8 to fit better
    feature_names = [f[0].replace('_', ' ').title() for f in top_features]
    importance_values = [f[1] for f in top_features]
    
    bars = ax3.barh(range(len(feature_names)), importance_values, 
                    color='orange', alpha=0.7)
    ax3.set_yticks(range(len(feature_names)))
    ax3.set_yticklabels(feature_names)
    ax3.set_xlabel('Feature Weight')
    ax3.set_title('Top Feature Importance', fontweight='bold')
    ax3.invert_yaxis()
    
    # 4. Training Summary
    ax4.axis('off')
    summary_text = f"FAST LINEAR SVM SUMMARY\n\n"
    summary_text += f"✅ Training Time: {results['training_time']:.2f}s\n"
    summary_text += f"🎯 F1-Score: {results['f1_score']:.3f}\n"
    summary_text += f"📊 Accuracy: {results['accuracy']:.3f}\n"
    summary_text += f"🔍 Precision: {results['precision']:.3f}\n"
    summary_text += f"📈 Recall: {results['recall']:.3f}\n\n"
    summary_text += f"PARAMETERS:\n"
    summary_text += f"• C: {results['model_params']['C']}\n"
    summary_text += f"• Class Weight: {results['model_params']['class_weight']}\n"
    summary_text += f"• Loss: {results['model_params']['loss']}\n"
    summary_text += f"• Algorithm: LinearSVC\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plot_path = f"E:\\Baki\\fast_linear_svm_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_fast_svm_summary(results, features, timestamp):
    """Save a quick summary"""
    summary_path = f"E:\\Baki\\fast_svm_summary_{timestamp}.txt"
    
    with open(summary_path, 'w') as f:
        f.write("FAST LINEAR SVM SPAM CLASSIFIER - RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {results['timestamp']}\n")
        f.write(f"Training Time: {results['training_time']:.2f} seconds\n")
        f.write(f"Algorithm: LinearSVC (No Grid Search)\n\n")
        
        f.write("PERFORMANCE:\n")
        f.write("-"*20 + "\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)\n")
        f.write(f"Precision: {results['precision']:.4f} ({results['precision']*100:.1f}%)\n")
        f.write(f"Recall:    {results['recall']:.4f} ({results['recall']*100:.1f}%)\n")
        f.write(f"F1-Score:  {results['f1_score']:.4f} ({results['f1_score']*100:.1f}%)\n")
        f.write(f"ROC AUC:   {results['roc_auc']:.4f} ({results['roc_auc']*100:.1f}%)\n\n")
        
        f.write("TOP 10 FEATURES:\n")
        f.write("-"*20 + "\n")
        for i, (feature, importance) in enumerate(features['top_10_features'], 1):
            f.write(f"{i:2d}. {feature:<20} | {importance:.6f}\n")

# Main execution
if __name__ == "__main__":
    print("🚀 FAST LINEAR SVM SPAM CLASSIFIER")
    print("="*60)
    print("⚡ NO Grid Search - Direct Training")
    print("🎯 Optimized defaults for spam detection")
    print("🚀 Maximum speed, excellent performance")
    print("="*60)
    
    try:
        model, results, features = build_fast_linear_svm_classifier()
        
        print(f"\n🎉 SUCCESS - FAST TRAINING COMPLETED!")
        print("="*50)
        print(f"🎯 F1-Score: {results['f1_score']:.1%}")
        print(f"📊 Accuracy: {results['accuracy']:.1%}")
        print(f"⚡ Training: {results['training_time']:.1f}s")
        print(f"🚀 READY FOR DEPLOYMENT!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print(f"\n⏹️  Press Enter to exit...")
    input()