import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler
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

def build_fast_nb_classifier(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """
    Fast Naive Bayes classifier with proper feature handling
    """
    print("🧠 FAST Naive Bayes Spam Classifier")
    print("="*42)
    print("⚡ Direct training - no grid search")
    print("🎯 Handles negative features properly")
    print("="*42)
    
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
        print("🔧 Filling missing values with 0...")
        X = X.fillna(0)
    
    # Check for negative values
    has_negative = (X < 0).any().any()
    print(f"🔍 Negative values detected: {has_negative}")
    
    # Split data
    print("✂️  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Split: {len(X_train):,} train, {len(X_test):,} test samples")
    
    # Prepare data for different NB variants
    print("🔧 Preparing data for NB variants...")
    
    # For MultinomialNB: make features non-negative using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # For BernoulliNB: binarize features (threshold at median)
    X_train_binary = (X_train > X_train.median()).astype(int)
    X_test_binary = (X_test > X_test.median()).astype(int)
    
    # Test NB variants with appropriate data
    models_data = {
        'Gaussian NB': {
            'model': GaussianNB(),
            'X_train': X_train,
            'X_test': X_test,
            'description': 'Handles continuous features, assumes normal distribution'
        },
        'Multinomial NB': {
            'model': MultinomialNB(alpha=1.0),
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'description': 'Uses scaled features (0-1 range)'
        },
        'Bernoulli NB': {
            'model': BernoulliNB(alpha=1.0),
            'X_train': X_train_binary,
            'X_test': X_test_binary,
            'description': 'Uses binary features (above/below median)'
        }
    }
    
    print("🚀 Training NB variants...")
    train_start = time.time()
    
    results = {}
    with tqdm(total=len(models_data), desc="🧠 Training Models", 
              bar_format="{desc}: {n_fmt}/{total_fmt}|{bar}| [{elapsed}]",
              ncols=80, colour='blue') as pbar:
        
        for name, config in models_data.items():
            model = config['model']
            X_tr = config['X_train']
            X_te = config['X_test']
            
            pbar.set_description(f"🧠 Training {name}")
            
            # Train model
            model.fit(X_tr, y_train)
            
            # Make predictions
            y_pred = model.predict(X_te)
            y_pred_proba = model.predict_proba(X_te)[:, 1]
            
            # Calculate metrics
            results[name] = {
                'model': model,
                'data_config': config,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'description': config['description']
            }
            
            pbar.update(1)
    
    train_time = time.time() - train_start
    print(f"✅ All models trained in {train_time:.2f} seconds")
    
    # Display results and select best
    print(f"\n📊 MODEL COMPARISON:")
    print("-" * 60)
    print(f"{'Model':<15} {'F1-Score':<10} {'Accuracy':<10} {'ROC AUC':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<15} {result['f1_score']:<10.4f} {result['accuracy']:<10.4f} {result['roc_auc']:<10.4f}")
    
    # Select best model based on F1 score
    best_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_model_info = results[best_name]
    best_model = best_model_info['model']
    
    print(f"\n🏆 BEST MODEL: {best_name}")
    print(f"   F1-Score: {best_model_info['f1_score']:.4f}")
    print(f"   Description: {best_model_info['description']}")
    
    # Cross-validation with proper data
    print(f"🔄 Running cross-validation...")
    cv_data = best_model_info['data_config']['X_train']
    cv_scores = cross_val_score(best_model, cv_data, y_train, cv=3, scoring='f1')
    
    # Prepare final results
    final_results = {
        'best_variant': best_name,
        'accuracy': best_model_info['accuracy'],
        'precision': best_model_info['precision'],
        'recall': best_model_info['recall'],
        'f1_score': best_model_info['f1_score'],
        'roc_auc': best_model_info['roc_auc'],
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'training_time': train_time,
        'all_variants': {name: {
            'f1': res['f1_score'], 
            'accuracy': res['accuracy'],
            'description': res['description']
        } for name, res in results.items()},
        'data_preprocessing': {
            'multinomial_scaled': 'MinMaxScaler applied',
            'bernoulli_binary': 'Binary threshold at median',
            'gaussian_original': 'Original continuous features'
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Feature insights
    feature_insights = get_nb_insights(best_model, feature_columns, best_name, best_model_info)
    
    # Print comprehensive results
    print(f"\n" + "="*60)
    print(f"🎉 TRAINING COMPLETE - PERFORMANCE SUMMARY")
    print(f"="*60)
    
    print(f"\n📊 BEST MODEL PERFORMANCE:")
    print("-" * 35)
    metrics_display = [
        ("🎯 Accuracy", final_results['accuracy']),
        ("🔍 Precision", final_results['precision']),
        ("📈 Recall", final_results['recall']),
        ("⚖️  F1-Score", final_results['f1_score']),
        ("📊 ROC AUC", final_results['roc_auc']),
        ("🔄 CV F1", final_results['cv_score'])
    ]
    
    for name, value in metrics_display:
        print(f"   {name:<15}: {value:.4f} ({value*100:.1f}%)")
    
    print(f"\n⏱️  TIMING SUMMARY:")
    print("-" * 25)
    print(f"   Data Loading: {load_time:.2f}s")
    print(f"   Model Training: {train_time:.2f}s")
    print(f"   Total Runtime: {load_time + train_time:.2f}s")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_model_info['y_pred'])
    print(f"\n🎯 CONFUSION MATRIX:")
    print("-" * 25)
    print(f"   True Negatives (Ham → Ham):   {cm[0,0]:,}")
    print(f"   False Positives (Ham → Spam): {cm[0,1]:,}")
    print(f"   False Negatives (Spam → Ham): {cm[1,0]:,}")
    print(f"   True Positives (Spam → Spam): {cm[1,1]:,}")
    
    # Save everything
    save_nb_results(best_model, final_results, feature_insights, y_test, 
                   best_model_info['y_pred'], best_model_info, scaler if best_name == 'Multinomial NB' else None)
    
    return best_model, final_results, feature_insights

def get_nb_insights(model, feature_columns, model_name, model_info):
    """Extract insights from the trained model"""
    insights = {
        'model_type': model_name,
        'n_features': len(feature_columns),
        'feature_names': feature_columns,
        'description': model_info['description']
    }
    
    if model_name == 'Gaussian NB':
        # Class means and variances
        insights['class_means'] = {
            'ham_class': model.theta_[0].tolist(),
            'spam_class': model.theta_[1].tolist()
        }
        insights['class_variances'] = {
            'ham_class': model.var_[0].tolist(),
            'spam_class': model.var_[1].tolist()
        }
        
        # Feature differences
        feature_diffs = []
        for i, feature in enumerate(feature_columns):
            diff = abs(model.theta_[1][i] - model.theta_[0][i])
            feature_diffs.append((feature, diff))
        
        feature_diffs.sort(key=lambda x: x[1], reverse=True)
        insights['top_discriminative_features'] = feature_diffs[:10]
    
    return insights

def save_nb_results(model, results, insights, y_test, y_pred, model_info, scaler=None):
    """Save all results with proper encoding"""
    base_path = "E:\\Baki\\"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n💾 SAVING RESULTS...")
    
    save_tasks = ["Model", "Results", "Insights", "Plot", "Summary"]
    
    with tqdm(total=len(save_tasks), desc="💾 Saving", 
              bar_format="{desc}: {n_fmt}/{total_fmt}|{bar}|",
              ncols=50) as pbar:
        
        # 1. Save model and scaler
        model_path = f"{base_path}fast_nb_model_{timestamp}.pkl"
        model_package = {
            'model': model,
            'scaler': scaler,
            'model_type': results['best_variant'],
            'feature_names': insights['feature_names']
        }
        joblib.dump(model_package, model_path)
        pbar.update(1)
        
        # 2. Save results with UTF-8 encoding
        results_path = f"{base_path}fast_nb_results_{timestamp}.json"
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = {
            'true_negatives': int(cm[0,0]),
            'false_positives': int(cm[0,1]),
            'false_negatives': int(cm[1,0]),
            'true_positives': int(cm[1,1])
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:  # Add encoding
            json.dump(results, f, indent=2, ensure_ascii=False)
        pbar.update(1)
        
        # 3. Save insights with UTF-8 encoding
        insights_path = f"{base_path}fast_nb_insights_{timestamp}.json"
        with open(insights_path, 'w', encoding='utf-8') as f:  # Add encoding
            json.dump(insights, f, indent=2, ensure_ascii=False)
        pbar.update(1)
        
        # 4. Create plot
        create_nb_plot(y_test, y_pred, results, insights, timestamp)
        pbar.update(1)
        
        # 5. Save summary
        save_nb_summary(results, insights, timestamp)
        pbar.update(1)
    
    print(f"\n✅ All files saved!")
    print(f"📁 Files saved with timestamp: {timestamp}")

def create_nb_plot(y_test, y_pred, results, insights, timestamp):
    """Create visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fast Naive Bayes - Spam Classification Results', fontsize=16, fontweight='bold')
    
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
    
    # 3. Model Comparison
    variants = list(results['all_variants'].keys())
    f1_scores = [results['all_variants'][v]['f1'] for v in variants]
    colors = ['gold' if v == results['best_variant'] else 'lightblue' for v in variants]
    
    bars = ax3.bar(variants, f1_scores, color=colors, alpha=0.8)
    ax3.set_title('NB Variants Comparison', fontweight='bold')
    ax3.set_ylabel('F1 Score')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Summary
    ax4.axis('off')
    summary_text = f"NAIVE BAYES SUMMARY\n\n"
    summary_text += f"🏆 Best: {results['best_variant']}\n"
    summary_text += f"⚡ Training: {results['training_time']:.1f}s\n"
    summary_text += f"🎯 F1-Score: {results['f1_score']:.3f}\n"
    summary_text += f"📊 Accuracy: {results['accuracy']:.3f}\n"
    summary_text += f"🔄 CV Score: {results['cv_score']:.3f}\n\n"
    summary_text += f"ADVANTAGES:\n"
    summary_text += f"• Fastest training\n"
    summary_text += f"• No hyperparameters\n"
    summary_text += f"• Interpretable\n"
    summary_text += f"• Good baseline\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plot_path = f"E:\\Baki\\fast_nb_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_nb_summary(results, insights, timestamp):
    """Save summary without Unicode characters"""
    summary_path = f"E:\\Baki\\fast_nb_summary_{timestamp}.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:  # Specify UTF-8 encoding
        f.write("FAST NAIVE BAYES SPAM CLASSIFIER - RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {results['timestamp']}\n")
        f.write(f"Training Time: {results['training_time']:.2f} seconds\n")
        f.write(f"Best Model: {results['best_variant']}\n\n")
        
        f.write("PERFORMANCE:\n")
        f.write("-"*20 + "\n")
        f.write(f"Accuracy:   {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)\n")
        f.write(f"Precision:  {results['precision']:.4f} ({results['precision']*100:.1f}%)\n")
        f.write(f"Recall:     {results['recall']:.4f} ({results['recall']*100:.1f}%)\n")
        f.write(f"F1-Score:   {results['f1_score']:.4f} ({results['f1_score']*100:.1f}%)\n")
        f.write(f"ROC AUC:    {results['roc_auc']:.4f} ({results['roc_auc']*100:.1f}%)\n")
        f.write(f"CV F1:      {results['cv_score']:.4f} +/- {results['cv_std']:.3f}\n\n")
        
        f.write("MODEL COMPARISON:\n")
        f.write("-"*20 + "\n")
        for variant, scores in results['all_variants'].items():
            marker = "BEST " if variant == results['best_variant'] else "     "  # Remove emoji
            f.write(f"{marker}{variant:<15} | F1: {scores['f1']:.4f}\n")
        
        f.write(f"\nNAIVE BAYES ADVANTAGES:\n")
        f.write("-"*25 + "\n")
        f.write("- Fastest training time\n")
        f.write("- No hyperparameter tuning needed\n")
        f.write("- Simple and interpretable\n")
        f.write("- Good baseline performance\n")
        f.write("- Handles irrelevant features well\n")
        f.write("- Excellent for real-time applications\n")

# Main execution
if __name__ == "__main__":
    print("🧠 FAST NAIVE BAYES SPAM CLASSIFIER")
    print("="*50)
    print("⚡ No grid search - optimized for speed")
    print("🔧 Proper handling of negative features")
    print("🎯 Tests all 3 NB variants automatically")
    print("="*50)
    
    try:
        model, results, insights = build_fast_nb_classifier()
        
        print(f"\n🎉 SUCCESS - FAST TRAINING COMPLETED!")
        print("="*50)
        print(f"🎯 F1-Score: {results['f1_score']:.1%}")
        print(f"📊 Accuracy: {results['accuracy']:.1%}")
        print(f"🏆 Best: {results['best_variant']}")
        print(f"⚡ Training: {results['training_time']:.1f}s")
        print(f"🚀 FASTEST ML ALGORITHM!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n⏹️  Press Enter to exit...")
    input()