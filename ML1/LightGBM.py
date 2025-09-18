import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def build_minimal_lgb_classifier(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """
    Minimal LightGBM spam classifier with essential saving functionality
    """
    print("⚡ Minimal LightGBM Spam Classifier")
    print("="*40)
    
    # Load data
    print("📁 Loading features...")
    df = pd.read_csv(csv_file)
    feature_columns = [col for col in df.columns if col != 'label']
    
    X = df[feature_columns]
    y = df['label']
    
    # Handle missing values (LightGBM handles them naturally, but let's be explicit)
    if X.isnull().sum().sum() > 0:
        print(f"⚠️ Found {X.isnull().sum().sum()} missing values (LightGBM will handle them)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data: {len(X_train)} train, {len(X_test)} test samples")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # LightGBM parameters (optimized for spam detection)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    print("🚀 Training LightGBM model...")
    
    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)  # Silent training
        ]
    )
    
    print(f"✅ Training completed with {model.num_trees()} trees")
    
    # Make predictions
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_iteration': model.best_iteration,
        'num_trees': model.num_trees(),
        'best_score': model.best_score['valid']['binary_logloss'],
        'params': params,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Get feature importance
    feature_importance = get_lgb_feature_importance(model, feature_columns)
    
    # Print results
    print(f"\n📊 Results:")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1_score']:.4f}")
    print(f"   ROC AUC:   {results['roc_auc']:.4f}")
    print(f"   Trees:     {results['num_trees']}")
    
    print(f"\n🎯 Top 5 Important Features:")
    for i, (feature, importance) in enumerate(feature_importance['top_10_features'][:5], 1):
        print(f"   {i}. {feature}: {importance}")
    
    # Save everything
    save_lgb_results(model, results, feature_importance, y_test, y_pred, y_pred_proba)
    
    return model, results, feature_importance

def get_lgb_feature_importance(model, feature_columns):
    """
    Extract feature importance from LightGBM model
    """
    # Get different types of importance
    gain_importance = model.feature_importance(importance_type='gain')
    split_importance = model.feature_importance(importance_type='split')
    
    # Create importance ranking based on gain (default and most meaningful)
    importance_pairs = list(zip(feature_columns, gain_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    feature_importance = {
        'gain_importance': {feature: int(imp) for feature, imp in importance_pairs},
        'split_importance': {feature: int(split_imp) for feature, split_imp in zip(feature_columns, split_importance)},
        'top_10_features': importance_pairs[:10],
        'total_gain': int(sum(gain_importance)),
        'total_splits': int(sum(split_importance))
    }
    
    return feature_importance

def save_lgb_results(model, results, features, y_test, y_pred, y_pred_proba):
    """
    Save LightGBM model and results to files
    """
    base_path = "E:\\Baki\\"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save the trained model (LightGBM native format)
    model_path = f"{base_path}lgb_model_{timestamp}.txt"
    model.save_model(model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # Also save as pickle for sklearn compatibility
    model_pkl_path = f"{base_path}lgb_model_{timestamp}.pkl"
    joblib.dump(model, model_pkl_path)
    print(f"💾 Model (pkl) saved: {model_pkl_path}")
    
    # 2. Save test results as JSON
    results_path = f"{base_path}lgb_results_{timestamp}.json"
    
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
    
    # 3. Save feature importance as JSON
    features_path = f"{base_path}lgb_features_{timestamp}.json"
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"💾 Features saved: {features_path}")
    
    # 4. Create performance plot
    create_lgb_plot(y_test, y_pred, y_pred_proba, results, features, timestamp)
    
    # 5. Save summary text file
    save_lgb_summary(results, features, timestamp)

def create_lgb_plot(y_test, y_pred, y_pred_proba, results, features, timestamp):
    """
    Create LightGBM performance visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LightGBM Spam Classifier Results', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ax1 = axes[0, 0]
    im = ax1.imshow(cm, interpolation='nearest', cmap='Greens')
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
    
    # 2. Performance Metrics
    ax2 = axes[0, 1]
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
    
    # 3. Feature Importance (Top 10)
    ax3 = axes[1, 0]
    top_features = features['top_10_features']
    feature_names = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    
    # Truncate long feature names for display
    short_names = [name[:12] + '...' if len(name) > 12 else name for name in feature_names]
    
    bars = ax3.barh(range(len(short_names)), importances, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(short_names))))
    ax3.set_yticks(range(len(short_names)))
    ax3.set_yticklabels(short_names, fontsize=9)
    ax3.set_xlabel('Feature Importance (Gain)')
    ax3.set_title('Top 10 Feature Importance')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Prediction Distribution
    ax4 = axes[1, 1]
    ham_probs = y_pred_proba[y_test == 0]
    spam_probs = y_pred_proba[y_test == 1]
    
    ax4.hist(ham_probs, bins=30, alpha=0.7, label=f'Ham (n={len(ham_probs)})', 
            color='green', density=True)
    ax4.hist(spam_probs, bins=30, alpha=0.7, label=f'Spam (n={len(spam_probs)})', 
            color='red', density=True)
    ax4.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax4.set_xlabel('Spam Probability')
    ax4.set_ylabel('Density')
    ax4.set_title('Prediction Distribution')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"E:\\Baki\\lgb_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved: {plot_path}")
    plt.show()

def save_lgb_summary(results, features, timestamp):
    """
    Save a readable LightGBM summary text file
    """
    summary_path = f"E:\\Baki\\lgb_summary_{timestamp}.txt"
    
    # Fix: Use UTF-8 encoding to handle Unicode characters
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("LIGHTGBM SPAM CLASSIFIER - MODEL SUMMARY\n")
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
        # Fix: Replace Unicode arrows with ASCII
        f.write(f"True Negatives (Ham->Ham):   {cm['true_negatives']}\n")
        f.write(f"False Positives (Ham->Spam): {cm['false_positives']}\n")
        f.write(f"False Negatives (Spam->Ham): {cm['false_negatives']}\n")
        f.write(f"True Positives (Spam->Spam): {cm['true_positives']}\n\n")
        
        f.write("LIGHTGBM MODEL CONFIGURATION:\n")
        f.write("-"*30 + "\n")
        f.write(f"Number of Trees: {results['num_trees']}\n")
        f.write(f"Best Iteration: {results['best_iteration']}\n")
        f.write(f"Best Score (Log Loss): {results['best_score']:.6f}\n")
        f.write(f"Learning Rate: {results['params']['learning_rate']}\n")
        f.write(f"Num Leaves: {results['params']['num_leaves']}\n")
        f.write(f"Feature Fraction: {results['params']['feature_fraction']}\n")
        f.write(f"Bagging Fraction: {results['params']['bagging_fraction']}\n\n")
        
        f.write("TOP 10 IMPORTANT FEATURES (by Gain):\n")
        f.write("-"*40 + "\n")
        for i, (feature, importance) in enumerate(features['top_10_features'], 1):
            percentage = (importance / features['total_gain']) * 100
            f.write(f"{i:2d}. {feature:<25} | Gain: {importance:,} ({percentage:.1f}%)\n")
        
        f.write(f"\nFEATURE IMPORTANCE SUMMARY:\n")
        f.write("-"*30 + "\n")
        f.write(f"Total Gain: {features['total_gain']:,}\n")
        f.write(f"Total Splits: {features['total_splits']:,}\n")
        top_5_gain = sum([f[1] for f in features['top_10_features'][:5]])
        f.write(f"Top 5 features account for: {(top_5_gain/features['total_gain'])*100:.1f}% of total gain\n")
        
        f.write(f"\nLIGHTGBM ADVANTAGES:\n")
        f.write("-"*20 + "\n")
        f.write("- Extremely fast training and prediction\n")
        f.write("- Handles missing values automatically\n")
        f.write("- Built-in overfitting protection\n")
        f.write("- Excellent feature importance\n")
        f.write("- Memory efficient\n")
        f.write("- Supports categorical features natively\n")
        f.write("- Early stopping prevents overfitting\n")
        
        f.write(f"\nMODEL CHARACTERISTICS:\n")
        f.write("-"*22 + "\n")
        f.write("- Gradient Boosting Decision Trees\n")
        f.write("- Leaf-wise tree growth (vs level-wise)\n")
        f.write("- Optimized for speed and memory\n")
        f.write("- Automatic feature selection via importance\n")
    
    print(f"Summary saved: {summary_path}")

# Main execution
if __name__ == "__main__":
    print("⚡ MINIMAL LIGHTGBM SPAM CLASSIFIER")
    print("="*42)
    
    try:
        # Check if LightGBM is installed
        import lightgbm as lgb
        print("✅ LightGBM is available")
        
        model, results, features = build_minimal_lgb_classifier()
        
        print(f"\n✅ TRAINING COMPLETE!")
        print("="*30)
        print(f"📊 F1-Score: {results['f1_score']:.1%}")
        print(f"🎯 Accuracy: {results['accuracy']:.1%}")
        print(f"⚡ Trees: {results['num_trees']}")
        print(f"🏆 Top feature: {features['top_10_features'][0][0]}")
        print(f"💾 All files saved with timestamp")
        
        # LightGBM advantages
        print(f"\n⚡ LightGBM Advantages:")
        print(f"   ✅ Fastest gradient boosting")
        print(f"   ✅ Handles missing values natively")
        print(f"   ✅ Built-in early stopping")
        print(f"   ✅ Memory efficient")
        print(f"   ✅ Excellent feature importance")
        print(f"   ✅ Often best performance")
        print(f"🚀 Ideal for production systems!")
        
    except ImportError:
        print("❌ Error: LightGBM not installed!")
        print("💡 Install with: pip install lightgbm")
    except FileNotFoundError:
        print("❌ Error: ml_features_matrix.csv not found!")
        print("💡 Please check the file path")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    input("\nPress Enter to exit...")