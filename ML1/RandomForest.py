import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
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

def build_minimal_rf_classifier(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """
    Minimal Random Forest spam classifier with essential saving functionality
    """
    print("🌲 Minimal Random Forest Spam Classifier")
    print("="*45)
    
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
    
    # Create pipeline options
    pipelines = {
        'rf_all_features': Pipeline([
            ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ]),
        'rf_selected_features': Pipeline([
            ('feature_selection', SelectKBest(f_classif, k=20)),
            ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
    }
    
    # Minimal parameter grids for faster training
    param_grids = {
        'rf_all_features': {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [20, None],
            'rf__min_samples_split': [2, 5],
            'rf__max_features': ['sqrt', 'log2'],
            'rf__class_weight': ['balanced']
        },
        'rf_selected_features': {
            'feature_selection__k': [15, 20, 25],
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [15, None],
            'rf__min_samples_split': [2, 5],
            'rf__max_features': ['sqrt'],
            'rf__class_weight': ['balanced']
        }
    }
    
    # Train both configurations
    print("🔍 Training models...")
    best_models = {}
    
    for name, pipeline in pipelines.items():
        print(f"   Training {name}...")
        rs = RandomizedSearchCV(
            pipeline, param_grids[name], 
            n_iter=12, cv=3, scoring='f1', 
            n_jobs=-1, random_state=42
        )
        rs.fit(X_train, y_train)
        best_models[name] = rs
    
    # Select best model
    model_scores = {name: model.best_score_ for name, model in best_models.items()}
    best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k])
    best_model = best_models[best_model_name].best_estimator_
    
    print(f"🏆 Best model: {best_model_name}")
    print(f"   CV F1-Score: {best_models[best_model_name].best_score_:.4f}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    results = {
        'model_type': best_model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_params': best_models[best_model_name].best_params_,
        'cv_score': best_models[best_model_name].best_score_,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Get feature importance and selection info
    important_features = get_rf_feature_importance(best_model, feature_columns)
    
    # Print results
    print(f"\n📊 Results:")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1_score']:.4f}")
    print(f"   ROC AUC:   {results['roc_auc']:.4f}")
    
    print(f"\n🎯 Top 5 Important Features:")
    for i, (feature, importance) in enumerate(important_features['top_10_features'][:5], 1):
        print(f"   {i}. {feature}: {importance:.4f}")
    
    # Save everything
    save_rf_results(best_model, results, important_features, y_test, y_pred)
    
    return best_model, results, important_features

def get_rf_feature_importance(model, feature_columns):
    """
    Extract feature importance from Random Forest model
    """
    # Handle pipeline structure
    if hasattr(model, 'named_steps'):
        if 'rf' in model.named_steps:
            rf_model = model.named_steps['rf']
            # Check if feature selection was used
            if 'feature_selection' in model.named_steps:
                selector = model.named_steps['feature_selection']
                selected_mask = selector.get_support()
                used_features = [feature_columns[i] for i, selected in enumerate(selected_mask) if selected]
                feature_scores = {feature_columns[i]: float(selector.scores_[i]) for i in range(len(feature_columns))}
            else:
                used_features = feature_columns
                feature_scores = {}
        else:
            rf_model = model
            used_features = feature_columns
            feature_scores = {}
    else:
        rf_model = model
        used_features = feature_columns
        feature_scores = {}
    
    # Get Random Forest feature importance
    rf_importance = rf_model.feature_importances_
    
    # Create importance ranking
    importance_pairs = list(zip(used_features, rf_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    important_features = {
        'used_features': used_features,
        'rf_importance': {feature: float(importance) for feature, importance in importance_pairs},
        'feature_selection_scores': feature_scores,
        'top_10_features': importance_pairs[:10],
        'model_config': {
            'n_estimators': rf_model.n_estimators,
            'max_depth': rf_model.max_depth,
            'min_samples_split': rf_model.min_samples_split,
            'min_samples_leaf': rf_model.min_samples_leaf,
            'max_features': rf_model.max_features,
            'bootstrap': rf_model.bootstrap
        }
    }
    
    return important_features

def save_rf_results(model, results, features, y_test, y_pred):
    """
    Save Random Forest model and results to files
    """
    base_path = "E:\\Baki\\"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save the trained model
    model_path = f"{base_path}rf_model_{timestamp}.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # 2. Save test results as JSON
    results_path = f"{base_path}rf_results_{timestamp}.json"
    
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
    features_path = f"{base_path}rf_features_{timestamp}.json"
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"💾 Features saved: {features_path}")
    
    # 4. Create simple performance plot
    create_rf_plot(y_test, y_pred, results, features, timestamp)
    
    # 5. Save summary text file
    save_rf_summary(results, features, timestamp)

def create_rf_plot(y_test, y_pred, results, features, timestamp):
    """
    Create simple Random Forest performance visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ax1 = axes[0]
    im = ax1.imshow(cm, interpolation='nearest', cmap='Greens')
    ax1.set_title('Confusion Matrix\n(Random Forest)')
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
    
    # Feature Importance (Top 10)
    ax3 = axes[2]
    top_features = features['top_10_features'][:10]
    feature_names = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    
    # Truncate long feature names
    short_names = [name[:15] + '...' if len(name) > 15 else name for name in feature_names]
    
    bars = ax3.barh(range(len(short_names)), importances, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(short_names))))
    ax3.set_yticks(range(len(short_names)))
    ax3.set_yticklabels(short_names, fontsize=9)
    ax3.set_xlabel('Importance')
    ax3.set_title('Top 10 Feature Importance')
    ax3.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"E:\\Baki\\rf_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved: {plot_path}")
    plt.show()

def save_rf_summary(results, features, timestamp):
    """
    Save a readable Random Forest summary text file
    """
    summary_path = f"E:\\Baki\\rf_summary_{timestamp}.txt"
    
    with open(summary_path, 'w') as f:
        f.write("RANDOM FOREST SPAM CLASSIFIER - MODEL SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Training Date: {results['timestamp']}\n")
        f.write(f"Model Type: {results['model_type']}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*25 + "\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)\n")
        f.write(f"Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)\n")
        f.write(f"F1-Score:  {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)\n")
        f.write(f"ROC AUC:   {results['roc_auc']:.4f} ({results['roc_auc']*100:.2f}%)\n")
        f.write(f"CV Score:  {results['cv_score']:.4f} ({results['cv_score']*100:.2f}%)\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-"*20 + "\n")
        cm = results['confusion_matrix']
        f.write(f"True Negatives (Ham→Ham):   {cm['true_negatives']}\n")
        f.write(f"False Positives (Ham→Spam): {cm['false_positives']}\n")
        f.write(f"False Negatives (Spam→Ham): {cm['false_negatives']}\n")
        f.write(f"True Positives (Spam→Spam): {cm['true_positives']}\n\n")
        
        f.write("RANDOM FOREST CONFIGURATION:\n")
        f.write("-"*30 + "\n")
        config = features['model_config']
        f.write(f"Trees (n_estimators): {config['n_estimators']}\n")
        f.write(f"Max Depth: {config['max_depth']}\n")
        f.write(f"Min Samples Split: {config['min_samples_split']}\n")
        f.write(f"Min Samples Leaf: {config['min_samples_leaf']}\n")
        f.write(f"Max Features: {config['max_features']}\n")
        f.write(f"Bootstrap: {config['bootstrap']}\n\n")
        
        f.write("BEST MODEL PARAMETERS:\n")
        f.write("-"*25 + "\n")
        for param, value in results['best_params'].items():
            f.write(f"{param}: {value}\n")
        
        f.write(f"\nTOP 10 IMPORTANT FEATURES:\n")
        f.write("-"*30 + "\n")
        for i, (feature, importance) in enumerate(features['top_10_features'], 1):
            f.write(f"{i:2d}. {feature:<25} | Importance: {importance:.4f}\n")
        
        f.write(f"\nFEATURE IMPORTANCE DISTRIBUTION:\n")
        f.write("-"*35 + "\n")
        importances = [f[1] for f in features['top_10_features']]
        f.write(f"Top 5 features capture: {sum(importances[:5]):.1%} of total importance\n")
        f.write(f"Top 10 features capture: {sum(importances):.1%} of total importance\n")
    
    print(f"📄 Summary saved: {summary_path}")

# Main execution
if __name__ == "__main__":
    print("🌲 MINIMAL RANDOM FOREST SPAM CLASSIFIER")
    print("="*50)
    
    try:
        model, results, features = build_minimal_rf_classifier()
        
        print(f"\n✅ TRAINING COMPLETE!")
        print("="*35)
        print(f"📊 F1-Score: {results['f1_score']:.1%}")
        print(f"🎯 Accuracy: {results['accuracy']:.1%}")
        print(f"🌲 Model: {results['model_type']}")
        print(f"🏆 Top feature: {features['top_10_features'][0][0]}")
        print(f"💾 All files saved with timestamp")
        print(f"🚀 Ready for deployment!")
        
        # Quick advantage summary
        print(f"\n🌲 Random Forest Advantages:")
        print(f"   ✅ No feature scaling needed")
        print(f"   ✅ Built-in feature importance")
        print(f"   ✅ Handles missing values well")
        print(f"   ✅ Resistant to overfitting")
        print(f"   ✅ Fast training and prediction")
        
    except FileNotFoundError:
        print("❌ Error: ml_features_matrix.csv not found!")
        print("💡 Please check the file path")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    input("\nPress Enter to exit...")