import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

def build_fast_rf_classifier(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """
    Fast Random Forest classifier - NO GRID SEARCH
    """
    print("🌲 FAST Random Forest Spam Classifier")
    print("="*50)
    print("⚡ NO Grid Search - Direct Training")
    print("🎯 Using optimized defaults")
    print("="*50)
    
    # Load data
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
    
    # Create fast Random Forest with good defaults
    print(f"\n🏗️  Building Fast Random Forest...")
    
    # Optimized parameters for speed and performance
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Good balance of performance vs speed
        max_depth=20,      # Prevent overfitting
        min_samples_split=5,  # Speed up training
        min_samples_leaf=2,   # Speed up training
        max_features='sqrt',  # Standard for classification
        class_weight='balanced',  # Handle any imbalance
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        bootstrap=True,
        oob_score=True  # Get out-of-bag score for free
    )
    
    print(f"⚙️  Using optimized parameters:")
    print(f"   • Trees: 100 (good speed/performance balance)")
    print(f"   • Max Depth: 20")
    print(f"   • Min Samples Split: 5")
    print(f"   • Max Features: sqrt")
    print(f"   • Class Weight: balanced")
    print(f"   • OOB Score: enabled")
    
    # Train with progress bar
    print(f"\n🚀 Training Random Forest...")
    print("=" * 40)
    
    train_start = time.time()
    
    with tqdm(total=100, desc="🌲 Training RF", 
              bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]",
              ncols=80, colour='green') as pbar:
        
        pbar.set_description("🌱 Growing trees")
        rf_model.fit(X_train, y_train)
        pbar.update(100)
        pbar.set_description("✅ Forest complete")
    
    train_time = time.time() - train_start
    print(f"✅ Training completed in {train_time:.2f} seconds!")
    
    # Make predictions
    print(f"\n🔮 Making predictions...")
    pred_start = time.time()
    
    with tqdm(total=100, desc="🎯 Predicting", ncols=60, colour='blue') as pbar:
        y_pred = rf_model.predict(X_test)
        pbar.update(50)
        
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
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
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'oob_score': rf_model.oob_score_,  # Out-of-bag score
        'training_time': train_time,
        'prediction_time': pred_time,
        'model_params': {
            'n_estimators': rf_model.n_estimators,
            'max_depth': rf_model.max_depth,
            'min_samples_split': rf_model.min_samples_split,
            'min_samples_leaf': rf_model.min_samples_leaf,
            'max_features': rf_model.max_features,
            'class_weight': rf_model.class_weight
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Feature importance
    print(f"🎯 Extracting feature importance...")
    feature_importance = rf_model.feature_importances_
    
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
        ("📊 ROC AUC", results['roc_auc']),
        ("🌳 OOB Score", results['oob_score'])
    ]
    
    for name, value in metrics_display:
        print(f"   {name:<15}: {value:.4f} ({value*100:.1f}%)")
    
    print(f"\n⚙️  MODEL CONFIGURATION:")
    print("-" * 30)
    for param, value in results['model_params'].items():
        print(f"   {param:<18}: {value}")
    
    print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
    print("-" * 50)
    for i, (feature, importance) in enumerate(feature_rankings['top_10_features'], 1):
        clean_feature = feature.replace('_', ' ').title()
        # Visual importance bar
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
    
    # Feature importance insights
    top_5_importance = sum([x[1] for x in feature_rankings['top_10_features'][:5]])
    print(f"\n🔍 FEATURE INSIGHTS:")
    print("-" * 25)
    print(f"   Top 5 features capture: {top_5_importance:.1%} of total importance")
    print(f"   Most important: {feature_rankings['top_10_features'][0][0]}")
    print(f"   Importance range: {feature_rankings['top_10_features'][-1][1]:.4f} to {feature_rankings['top_10_features'][0][1]:.4f}")
    
    # Confusion matrix details
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🎯 CONFUSION MATRIX:")
    print("-" * 25)
    print(f"   True Negatives (Ham → Ham):   {cm[0,0]:,}")
    print(f"   False Positives (Ham → Spam): {cm[0,1]:,}")
    print(f"   False Negatives (Spam → Ham): {cm[1,0]:,}")
    print(f"   True Positives (Spam → Spam): {cm[1,1]:,}")
    
    # Save everything
    save_fast_rf_results(rf_model, results, feature_rankings, y_test, y_pred, feature_columns)
    
    return rf_model, results, feature_rankings

def save_fast_rf_results(model, results, features, y_test, y_pred, feature_columns):
    """Save results with progress indication"""
    base_path = "E:\\Baki\\"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n💾 SAVING RESULTS...")
    
    save_tasks = ["Model", "Results", "Features", "Plots", "Summary"]
    
    with tqdm(total=len(save_tasks), desc="💾 Saving Files", 
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
              ncols=60, colour='cyan') as pbar:
        
        # 1. Save model
        model_path = f"{base_path}fast_rf_model_{timestamp}.pkl"
        joblib.dump(model, model_path)
        pbar.update(1)
        
        # 2. Save results
        results_path = f"{base_path}fast_rf_results_{timestamp}.json"
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
        features_path = f"{base_path}fast_rf_features_{timestamp}.json"
        with open(features_path, 'w') as f:
            json.dump(features, f, indent=2)
        pbar.update(1)
        
        # 4. Create plots
        create_fast_rf_plot(y_test, y_pred, results, features, timestamp)
        pbar.update(1)
        
        # 5. Summary
        save_fast_rf_summary(results, features, timestamp)
        pbar.update(1)
    
    print(f"\n✅ All files saved!")
    print(f"📁 Files saved with timestamp: {timestamp}")

def create_fast_rf_plot(y_test, y_pred, results, features, timestamp):
    """Create visualization for Random Forest results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fast Random Forest - Spam Classification Results', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    im = ax1.imshow(cm, interpolation='nearest', cmap='Greens')
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
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'OOB Score']
    values = [results['accuracy'], results['precision'], results['recall'], 
              results['f1_score'], results['roc_auc'], results['oob_score']]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
    ax2.set_title('Performance Metrics', fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Feature Importance
    top_features = features['top_10_features'][:10]
    feature_names = [f[0].replace('_', ' ').title() for f in top_features]
    importance_values = [f[1] for f in top_features]
    
    bars = ax3.barh(range(len(feature_names)), importance_values, 
                    color='forestgreen', alpha=0.7)
    ax3.set_yticks(range(len(feature_names)))
    ax3.set_yticklabels(feature_names, fontsize=9)
    ax3.set_xlabel('Feature Importance')
    ax3.set_title('Top 10 Feature Importance', fontweight='bold')
    ax3.invert_yaxis()
    
    # 4. Training Summary
    ax4.axis('off')
    summary_text = f"FAST RANDOM FOREST SUMMARY\n\n"
    summary_text += f"🌲 Trees: {results['model_params']['n_estimators']}\n"
    summary_text += f"⏱️  Training: {results['training_time']:.1f}s\n"
    summary_text += f"🎯 F1-Score: {results['f1_score']:.3f}\n"
    summary_text += f"📊 Accuracy: {results['accuracy']:.3f}\n"
    summary_text += f"🌳 OOB Score: {results['oob_score']:.3f}\n\n"
    summary_text += f"CONFIGURATION:\n"
    summary_text += f"• Max Depth: {results['model_params']['max_depth']}\n"
    summary_text += f"• Min Samples Split: {results['model_params']['min_samples_split']}\n"
    summary_text += f"• Max Features: {results['model_params']['max_features']}\n"
    summary_text += f"• Class Weight: {results['model_params']['class_weight']}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plot_path = f"E:\\Baki\\fast_rf_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_fast_rf_summary(results, features, timestamp):
    """Save summary report"""
    summary_path = f"E:\\Baki\\fast_rf_summary_{timestamp}.txt"
    
    with open(summary_path, 'w') as f:
        f.write("FAST RANDOM FOREST SPAM CLASSIFIER - RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {results['timestamp']}\n")
        f.write(f"Training Time: {results['training_time']:.2f} seconds\n")
        f.write(f"Algorithm: Random Forest (No Grid Search)\n\n")
        
        f.write("PERFORMANCE:\n")
        f.write("-"*20 + "\n")
        f.write(f"Accuracy:   {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)\n")
        f.write(f"Precision:  {results['precision']:.4f} ({results['precision']*100:.1f}%)\n")
        f.write(f"Recall:     {results['recall']:.4f} ({results['recall']*100:.1f}%)\n")
        f.write(f"F1-Score:   {results['f1_score']:.4f} ({results['f1_score']*100:.1f}%)\n")
        f.write(f"ROC AUC:    {results['roc_auc']:.4f} ({results['roc_auc']*100:.1f}%)\n")
        f.write(f"OOB Score:  {results['oob_score']:.4f} ({results['oob_score']*100:.1f}%)\n\n")
        
        f.write("TOP 10 FEATURE IMPORTANCE:\n")
        f.write("-"*30 + "\n")
        for i, (feature, importance) in enumerate(features['top_10_features'], 1):
            f.write(f"{i:2d}. {feature:<25} | {importance:.6f}\n")

# Main execution
if __name__ == "__main__":
    print("🌲 FAST RANDOM FOREST SPAM CLASSIFIER")
    print("="*60)
    print("⚡ NO Grid Search - Direct Training")
    print("🎯 Optimized defaults for maximum speed")
    print("🌲 100 trees, balanced classes, all cores")
    print("="*60)
    
    try:
        model, results, features = build_fast_rf_classifier()
        
        print(f"\n🎉 SUCCESS - FAST TRAINING COMPLETED!")
        print("="*50)
        print(f"🎯 F1-Score: {results['f1_score']:.1%}")
        print(f"📊 Accuracy: {results['accuracy']:.1%}")
        print(f"🌳 OOB Score: {results['oob_score']:.1%}")
        print(f"⚡ Training: {results['training_time']:.1f}s")
        print(f"🏆 Top Feature: {features['top_10_features'][0][0]}")
        print(f"🚀 READY FOR DEPLOYMENT!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print(f"\n⏹️  Press Enter to exit...")
    input()