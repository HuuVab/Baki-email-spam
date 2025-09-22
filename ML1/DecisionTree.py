import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def build_minimal_dt_classifier(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """
    Minimal Decision Tree spam classifier with essential saving functionality
    """
    print("🌳 Minimal Decision Tree Spam Classifier")
    print("="*42)
    
    # Load data
    print("📁 Loading features...")
    df = pd.read_csv(csv_file)
    feature_columns = [col for col in df.columns if col != 'label']
    
    X = df[feature_columns]
    y = df['label']
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        print(f"⚠️ Found {X.isnull().sum().sum()} missing values, filling with median")
        X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data: {len(X_train)} train, {len(X_test)} test samples")
    
    # Decision Tree parameters (optimized for spam detection)
    dt_params = {
        'criterion': 'gini',
        'max_depth': 10,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'random_state': 42,
        'class_weight': 'balanced'
    }
    
    print("🌳 Training Decision Tree model...")
    
    # Train Decision Tree model
    model = DecisionTreeClassifier(**dt_params)
    model.fit(X_train, y_train)
    
    print(f"✅ Training completed with {model.tree_.node_count} nodes")
    print(f"🍃 Tree depth: {model.tree_.max_depth}")
    print(f"🌿 Leaves: {model.tree_.n_leaves}")
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of spam class
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'tree_depth': int(model.tree_.max_depth),
        'tree_nodes': int(model.tree_.node_count),
        'tree_leaves': int(model.tree_.n_leaves),
        'params': dt_params,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Get feature importance
    feature_importance = get_dt_feature_importance(model, feature_columns)
    
    # Print results
    print(f"\n📊 Results:")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1_score']:.4f}")
    print(f"   ROC AUC:   {results['roc_auc']:.4f}")
    print(f"   Nodes:     {results['tree_nodes']}")
    print(f"   Leaves:    {results['tree_leaves']}")
    
    print(f"\n🎯 Top 5 Important Features:")
    for i, (feature, importance) in enumerate(feature_importance['top_10_features'][:5], 1):
        print(f"   {i}. {feature}: {importance:.4f}")
    
    # Save everything
    save_dt_results(model, results, feature_importance, y_test, y_pred, y_pred_proba, feature_columns)
    
    return model, results, feature_importance

def get_dt_feature_importance(model, feature_columns):
    """
    Extract feature importance from Decision Tree model
    """
    # Get feature importances (Gini importance)
    importances = model.feature_importances_
    
    # Create importance ranking
    importance_pairs = list(zip(feature_columns, importances))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Filter out zero importance features
    non_zero_importance = [(name, imp) for name, imp in importance_pairs if imp > 0]
    
    feature_importance = {
        'gini_importance': {feature: float(imp) for feature, imp in importance_pairs},
        'top_10_features': importance_pairs[:10],
        'used_features': len(non_zero_importance),
        'total_features': len(feature_columns),
        'max_importance': float(max(importances)) if len(importances) > 0 else 0
    }
    
    return feature_importance

def save_dt_results(model, results, features, y_test, y_pred, y_pred_proba, feature_columns):
    """
    Save Decision Tree model and results to files
    """
    base_path = "E:\\Baki\\"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save the trained model
    model_pkl_path = f"{base_path}dt_model_{timestamp}.pkl"
    joblib.dump(model, model_pkl_path)
    print(f"\n💾 Model saved: {model_pkl_path}")
    
    # 2. Save test results as JSON
    results_path = f"{base_path}dt_results_{timestamp}.json"
    
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
    features_path = f"{base_path}dt_features_{timestamp}.json"
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=2)
    print(f"💾 Features saved: {features_path}")
    
    # 4. Save tree rules as text
    save_tree_rules(model, feature_columns, timestamp)
    
    # 5. Create performance plot
    create_dt_plot(y_test, y_pred, y_pred_proba, results, features, timestamp)
    
    # 6. Save summary text file
    save_dt_summary(results, features, timestamp)

def save_tree_rules(model, feature_columns, timestamp):
    """
    Save decision tree rules as readable text
    """
    rules_path = f"E:\\Baki\\dt_rules_{timestamp}.txt"
    
    with open(rules_path, 'w', encoding='utf-8') as f:
        f.write("DECISION TREE RULES\n")
        f.write("="*30 + "\n\n")
        
        # Export tree structure as text
        tree_rules = export_text(model, feature_names=feature_columns, max_depth=5)
        f.write("DECISION RULES (Top 5 levels):\n")
        f.write("-"*35 + "\n")
        f.write(tree_rules)
        
        f.write("\n\nTREE INTERPRETATION:\n")
        f.write("-"*20 + "\n")
        f.write("- Each line shows a decision rule\n")
        f.write("- |--- indicates tree depth/level\n")
        f.write("- <= means 'less than or equal to'\n")
        f.write("- class: 0 = Ham (legitimate email)\n")
        f.write("- class: 1 = Spam (unwanted email)\n")
        f.write("- value: [ham_samples, spam_samples]\n")
    
    print(f"🌳 Tree rules saved: {rules_path}")

def create_dt_plot(y_test, y_pred, y_pred_proba, results, features, timestamp):
    """
    Create Decision Tree performance visualization
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create a more complex layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    fig.suptitle('Decision Tree Spam Classifier Results', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, y_pred)
    im = ax1.imshow(cm, interpolation='nearest', cmap='Oranges')
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
    ax2 = fig.add_subplot(gs[0, 1])
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
    
    # 3. Tree Statistics
    ax3 = fig.add_subplot(gs[0, 2])
    tree_stats = ['Depth', 'Nodes', 'Leaves', 'Features Used']
    tree_values = [results['tree_depth'], results['tree_nodes'], 
                   results['tree_leaves'], features['used_features']]
    
    bars = ax3.bar(tree_stats, tree_values, color=['brown', 'green', 'orange', 'purple'])
    ax3.set_title('Tree Structure')
    ax3.set_ylabel('Count')
    
    # Add value labels on bars
    for bar, value in zip(bars, tree_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Feature Importance (Top 10)
    ax4 = fig.add_subplot(gs[1, :])
    top_features = features['top_10_features']
    feature_names = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    
    # Truncate long feature names for display
    short_names = [name[:15] + '...' if len(name) > 15 else name for name in feature_names]
    
    bars = ax4.barh(range(len(short_names)), importances, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(short_names))))
    ax4.set_yticks(range(len(short_names)))
    ax4.set_yticklabels(short_names, fontsize=9)
    ax4.set_xlabel('Feature Importance (Gini)')
    ax4.set_title('Top 10 Feature Importance')
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Prediction Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    ham_probs = y_pred_proba[y_test == 0]
    spam_probs = y_pred_proba[y_test == 1]
    
    ax5.hist(ham_probs, bins=20, alpha=0.7, label=f'Ham (n={len(ham_probs)})', 
            color='green', density=True)
    ax5.hist(spam_probs, bins=20, alpha=0.7, label=f'Spam (n={len(spam_probs)})', 
            color='red', density=True)
    ax5.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax5.set_xlabel('Spam Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Prediction Distribution')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Feature Usage Pie Chart
    ax6 = fig.add_subplot(gs[2, 1])
    used_features = features['used_features']
    unused_features = features['total_features'] - used_features
    
    sizes = [used_features, unused_features]
    labels = [f'Used\n({used_features})', f'Unused\n({unused_features})']
    colors = ['lightgreen', 'lightcoral']
    
    ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Feature Usage')
    
    # 7. Tree Complexity
    ax7 = fig.add_subplot(gs[2, 2])
    complexity_labels = ['Max Depth', 'Min Samples\nSplit', 'Min Samples\nLeaf']
    complexity_values = [results['params']['max_depth'], 
                        results['params']['min_samples_split'],
                        results['params']['min_samples_leaf']]
    
    bars = ax7.bar(complexity_labels, complexity_values, color=['orange', 'purple', 'cyan'])
    ax7.set_title('Tree Parameters')
    ax7.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, complexity_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_path = f"E:\\Baki\\dt_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved: {plot_path}")
    plt.show()

def save_dt_summary(results, features, timestamp):
    """
    Save a readable Decision Tree summary text file
    """
    summary_path = f"E:\\Baki\\dt_summary_{timestamp}.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("DECISION TREE SPAM CLASSIFIER - MODEL SUMMARY\n")
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
        f.write(f"True Negatives (Ham->Ham):   {cm['true_negatives']}\n")
        f.write(f"False Positives (Ham->Spam): {cm['false_positives']}\n")
        f.write(f"False Negatives (Spam->Ham): {cm['false_negatives']}\n")
        f.write(f"True Positives (Spam->Spam): {cm['true_positives']}\n\n")
        
        f.write("DECISION TREE STRUCTURE:\n")
        f.write("-"*28 + "\n")
        f.write(f"Tree Depth: {results['tree_depth']}\n")
        f.write(f"Total Nodes: {results['tree_nodes']}\n")
        f.write(f"Leaf Nodes: {results['tree_leaves']}\n")
        f.write(f"Internal Nodes: {results['tree_nodes'] - results['tree_leaves']}\n")
        f.write(f"Features Used: {features['used_features']} / {features['total_features']}\n")
        f.write(f"Feature Usage: {(features['used_features']/features['total_features'])*100:.1f}%\n\n")
        
        f.write("DECISION TREE PARAMETERS:\n")
        f.write("-"*27 + "\n")
        f.write(f"Criterion: {results['params']['criterion']}\n")
        f.write(f"Max Depth: {results['params']['max_depth']}\n")
        f.write(f"Min Samples Split: {results['params']['min_samples_split']}\n")
        f.write(f"Min Samples Leaf: {results['params']['min_samples_leaf']}\n")
        f.write(f"Max Features: {results['params']['max_features']}\n")
        f.write(f"Class Weight: {results['params']['class_weight']}\n\n")
        
        f.write("TOP 10 IMPORTANT FEATURES (by Gini):\n")
        f.write("-"*40 + "\n")
        for i, (feature, importance) in enumerate(features['top_10_features'], 1):
            percentage = (importance / features['max_importance']) * 100 if features['max_importance'] > 0 else 0
            f.write(f"{i:2d}. {feature:<25} | Gini: {importance:.4f} ({percentage:.1f}%)\n")
        
        f.write(f"\nDECISION TREE ADVANTAGES:\n")
        f.write("-"*25 + "\n")
        f.write("- Highly interpretable and explainable\n")
        f.write("- No assumptions about data distribution\n")
        f.write("- Handles both numerical and categorical data\n")
        f.write("- Automatic feature selection\n")
        f.write("- Fast prediction\n")
        f.write("- Can capture non-linear relationships\n")
        f.write("- No need for feature scaling\n")
        f.write("- Provides clear decision rules\n")
        
        f.write(f"\nDECISION TREE CHARACTERISTICS:\n")
        f.write("-"*30 + "\n")
        f.write("- Recursive binary splitting\n")
        f.write("- Greedy algorithm (locally optimal)\n")
        f.write("- Prone to overfitting if not controlled\n")
        f.write("- Bias towards features with more levels\n")
        f.write("- Can be unstable (high variance)\n")
        f.write("- Works well with balanced classes\n")
        
        f.write(f"\nINTERPRETATION GUIDE:\n")
        f.write("-"*18 + "\n")
        f.write("- Check tree rules file for decision paths\n")
        f.write("- High Gini importance = more discriminative\n")
        f.write("- Shallow trees = simpler, more generalizable\n")
        f.write("- Deep trees = complex, potentially overfitted\n")
        f.write("- Few leaves = simpler model\n")
        f.write("- Many leaves = complex model\n")
    
    print(f"📄 Summary saved: {summary_path}")

# Main execution
if __name__ == "__main__":
    print("🌳 MINIMAL DECISION TREE SPAM CLASSIFIER")
    print("="*44)
    
    try:
        model, results, features = build_minimal_dt_classifier()
        
        print(f"\n✅ TRAINING COMPLETE!")
        print("="*30)
        print(f"📊 F1-Score: {results['f1_score']:.1%}")
        print(f"🎯 Accuracy: {results['accuracy']:.1%}")
        print(f"🌳 Depth: {results['tree_depth']}")
        print(f"🍃 Nodes: {results['tree_nodes']}")
        print(f"🏆 Top feature: {features['top_10_features'][0][0]}")
        print(f"💾 All files saved with timestamp")
        
        # Decision Tree advantages
        print(f"\n🌳 Decision Tree Advantages:")
        print(f"   ✅ Highly interpretable")
        print(f"   ✅ No data assumptions")
        print(f"   ✅ Handles mixed data types")
        print(f"   ✅ Fast predictions")
        print(f"   ✅ Clear decision rules")
        print(f"   ✅ No feature scaling needed")
        print(f"🔍 Perfect for understanding decisions!")
        
    except FileNotFoundError:
        print("❌ Error: ml_features_matrix.csv not found!")
        print("💡 Please check the file path")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    input("\nPress Enter to exit...")