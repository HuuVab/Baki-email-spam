import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

def build_svm_from_features_csv(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """
    Build SVM spam classifier using pre-computed features CSV
    """
    print("🚀 SVM SPAM DETECTION MODEL (Using Pre-computed Features)")
    print("="*65)
    
    # Load pre-computed features
    print("📁 Loading pre-computed features matrix...")
    df = pd.read_csv(csv_file)
    
    print(f"📊 Dataset size: {len(df):,} emails")
    print(f"📋 Total columns: {len(df.columns)}")
    print(f"🎯 Feature columns: {len(df.columns)-1} (excluding 'label')")
    
    # Display feature names
    feature_columns = [col for col in df.columns if col != 'label']
    print(f"\n📋 Available features:")
    for i, feature in enumerate(feature_columns, 1):
        print(f"   {i:2d}. {feature}")
    
    # Check label distribution
    label_counts = df['label'].value_counts().sort_index()
    print(f"\n📈 Label distribution:")
    for label, count in label_counts.items():
        label_name = "Ham (Legitimate)" if label == 0 else "Spam (Malicious)"
        percentage = (count / len(df)) * 100
        print(f"   • {label} ({label_name}): {count:,} emails ({percentage:.1f}%)")
    
    # Separate features and target
    X = df[feature_columns]
    y = df['label']
    
    # Check for missing values
    missing_values = X.isnull().sum().sum()
    if missing_values > 0:
        print(f"⚠️  Found {missing_values} missing values. Filling with median...")
        X = X.fillna(X.median())
    else:
        print("✅ No missing values found")
    
    # Split the data
    print(f"\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {len(X_train):,} emails")
    print(f"   Test set: {len(X_test):,} emails")
    
    # Analyze feature correlations with target
    print(f"\n🔍 Analyzing feature importance...")
    feature_correlations = []
    for feature in feature_columns:
        corr = df[feature].corr(df['label'])
        feature_correlations.append((feature, abs(corr), corr))
    
    # Sort by absolute correlation
    feature_correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Top 10 features by correlation with spam:")
    for i, (feature, abs_corr, corr) in enumerate(feature_correlations[:10], 1):
        direction = "↑ Spam" if corr > 0 else "↓ Ham"
        print(f"   {i:2d}. {feature:<25} | {abs_corr:.4f} | {direction}")
    
    # Create preprocessing pipeline
    print(f"\n⚙️ Creating preprocessing pipeline...")
    
    preprocessor = Pipeline([
        ('scaler', RobustScaler()),
        ('feature_selection', SelectKBest(f_classif, k='all'))  # Start with all, will tune
    ])
    
    # Create SVM pipeline
    svm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    # Comprehensive hyperparameter tuning
    print(f"🎛️ Setting up hyperparameter grid...")
    
    param_grid = [
        {
            'preprocessor__feature_selection__k': [10, 15, 20, 25],
            'svm__C': [0.1, 1, 10, 100],
            'svm__kernel': ['linear'],
            'svm__class_weight': [None, 'balanced']
        },
        {
            'preprocessor__feature_selection__k': [10, 15, 20],
            'svm__C': [0.1, 1, 10, 100],
            'svm__kernel': ['rbf'],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01],
            'svm__class_weight': [None, 'balanced']
        }
    ]
    
    # Grid search with cross-validation
    print(f"🔍 Performing Grid Search with 5-fold CV...")
    print(f"   Total combinations to test: ~{len(param_grid[0])*4 + len(param_grid[1])*4*4*2}")
    
    grid_search = GridSearchCV(
        svm_pipeline, param_grid, cv=5, 
        scoring='f1', n_jobs=-1, verbose=1
    )
    
    print(f"🚀 Training SVM models...")
    grid_search.fit(X_train, y_train)
    
    # Best model results
    best_svm = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print(f"\n✅ BEST MODEL FOUND:")
    print("="*40)
    print(f"Cross-validation F1 score: {best_cv_score:.4f}")
    print(f"Best parameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # Make predictions on test set
    print(f"\n🎯 Making predictions on test set...")
    y_pred = best_svm.predict(X_test)
    y_pred_proba = best_svm.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print detailed results
    print(f"\n📊 FINAL MODEL PERFORMANCE:")
    print("="*45)
    print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🎯 Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"🎯 F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"🎯 ROC AUC:   {auc:.4f} ({auc*100:.2f}%)")
    
    # Confusion matrix details
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📋 CONFUSION MATRIX BREAKDOWN:")
    print("-" * 40)
    print(f"   True Negatives (Ham → Ham):   {tn:,}")
    print(f"   False Positives (Ham → Spam): {fp:,}")
    print(f"   False Negatives (Spam → Ham): {fn:,}")
    print(f"   True Positives (Spam → Spam): {tp:,}")
    
    print(f"\n📊 CLASSIFICATION REPORT:")
    print("-" * 50)
    report = classification_report(y_test, y_pred, 
                                 target_names=['Ham (Legitimate)', 'Spam (Malicious)'],
                                 digits=4)
    print(report)
    
    # Cross-validation consistency check
    print(f"\n🔄 CROSS-VALIDATION CONSISTENCY:")
    print("-" * 45)
    cv_scores = cross_val_score(best_svm, X_train, y_train, cv=5, scoring='f1')
    print(f"F1 Scores across 5 folds: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Standard deviation: {cv_scores.std():.4f}")
    
    # Create comprehensive visualizations
    create_comprehensive_svm_plots(y_test, y_pred, y_pred_proba, best_svm, X_train, y_train, 
                                  feature_columns, feature_correlations)
    
    # Analyze selected features
    analyze_selected_features(best_svm, feature_columns, X_train, y_train)
    
    # Save the model
    model_path = "E:\\Baki\\svm_spam_classifier_final.pkl"
    joblib.dump(best_svm, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    return best_svm, X_test, y_test, y_pred, y_pred_proba, feature_columns

def create_comprehensive_svm_plots(y_test, y_pred, y_pred_proba, model, X_train, y_train, 
                                  feature_columns, feature_correlations):
    """
    Create comprehensive SVM visualization suite
    """
    print(f"\n🎨 Creating comprehensive visualizations...")
    
    # Set up the main plot
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('SVM Spam Detection Model - Complete Analysis Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Confusion Matrix with detailed annotations
    ax = axes[0, 0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='red')
    
    # 2. ROC Curve with confidence interval
    ax = axes[0, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontweight='bold', fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax = axes[0, 2]
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    ax.plot(recall_vals, precision_vals, color='red', lw=3, 
            label=f'AP = {avg_precision:.4f}')
    ax.fill_between(recall_vals, precision_vals, alpha=0.2, color='red')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Feature Importance (Top 15)
    ax = axes[1, 0]
    top_features = feature_correlations[:15]
    feature_names = [f[0] for f in top_features]
    correlations = [f[2] for f in top_features]  # Use signed correlation
    
    colors = ['red' if corr > 0 else 'blue' for corr in correlations]
    bars = ax.barh(range(len(feature_names)), correlations, color=colors, alpha=0.7)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel('Correlation with Spam')
    ax.set_title('Top 15 Features by Correlation\n(Red=Spam, Blue=Ham)', 
                fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    
    # 5. Prediction Confidence Distribution
    ax = axes[1, 1]
    ham_probs = y_pred_proba[y_test == 0]
    spam_probs = y_pred_proba[y_test == 1]
    
    ax.hist(ham_probs, bins=30, alpha=0.7, label=f'Ham (n={len(ham_probs)})', 
           color='green', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(spam_probs, bins=30, alpha=0.7, label=f'Spam (n={len(spam_probs)})', 
           color='red', density=True, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Threshold')
    ax.set_xlabel('Spam Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Confidence Distribution', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 6. Performance Metrics Radar Chart (converted to bar for simplicity)
    ax = axes[1, 2]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    scores = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        auc
    ]
    
    colors_metrics = ['skyblue', 'lightgreen', 'orange', 'pink', 'gold']
    bars = ax.bar(metrics, scores, color=colors_metrics, alpha=0.8, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_title('Performance Metrics Summary', fontweight='bold', fontsize=14)
    ax.set_ylabel('Score')
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 7. Learning Curve
    ax = axes[2, 0]
    from sklearn.model_selection import learning_curve
    
    # Use subset for faster computation
    sample_size = min(20000, len(X_train))
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train.iloc[:sample_size], y_train.iloc[:sample_size], 
        cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training F1', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                   alpha=0.1, color='blue')
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation F1', linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                   alpha=0.1, color='red')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('F1 Score')
    ax.set_title('Learning Curve', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 8. Threshold Analysis
    ax = axes[2, 1]
    thresholds_range = np.linspace(0.1, 0.9, 50)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for threshold in thresholds_range:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        precision_scores.append(precision_score(y_test, y_pred_thresh))
        recall_scores.append(recall_score(y_test, y_pred_thresh))
        f1_scores.append(f1_score(y_test, y_pred_thresh))
    
    ax.plot(thresholds_range, precision_scores, 'b-', label='Precision', linewidth=2)
    ax.plot(thresholds_range, recall_scores, 'r-', label='Recall', linewidth=2)
    ax.plot(thresholds_range, f1_scores, 'g-', label='F1-Score', linewidth=2)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Default (0.5)')
    
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold vs Performance', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 9. Model Summary Statistics
    ax = axes[2, 2]
    ax.axis('off')
    
    # Calculate additional metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)  # Negative Predictive Value
    
    summary_text = f"""
MODEL SUMMARY STATISTICS

Dataset Size: {len(y_test):,} test samples
Class Balance: {np.sum(y_test==0):,} Ham, {np.sum(y_test==1):,} Spam

PERFORMANCE METRICS:
- Accuracy:    {accuracy_score(y_test, y_pred):.4f}
- Precision:   {precision_score(y_test, y_pred):.4f}
- Recall:      {recall_score(y_test, y_pred):.4f}
- Specificity: {specificity:.4f}
- F1-Score:    {f1_score(y_test, y_pred):.4f}
- AUC-ROC:     {auc:.4f}
- NPV:         {npv:.4f}

CONFUSION MATRIX:
- True Positives:  {tp:,}
- True Negatives:  {tn:,}
- False Positives: {fp:,}
- False Negatives: {fn:,}

KERNEL: {model.named_steps['svm'].kernel.upper()}
FEATURES: {model.named_steps['preprocessor'].named_steps['feature_selection'].k}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("E:\\Baki\\svm_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    print("💾 Comprehensive SVM analysis saved: E:\\Baki\\svm_comprehensive_analysis.png")
    plt.show()

def analyze_selected_features(model, all_features, X_train, y_train):
    """
    Analyze which features were selected by the model
    """
    print(f"\n🎯 SELECTED FEATURES ANALYSIS:")
    print("="*50)
    
    # Get the feature selector
    feature_selector = model.named_steps['preprocessor'].named_steps['feature_selection']
    
    # Get selected features
    selected_mask = feature_selector.get_support()
    selected_features = [all_features[i] for i, selected in enumerate(selected_mask) if selected]
    
    print(f"📊 Selected {len(selected_features)} out of {len(all_features)} features:")
    
    # Get feature scores
    feature_scores = feature_selector.scores_
    selected_scores = [(all_features[i], feature_scores[i]) for i, selected in enumerate(selected_mask) if selected]
    selected_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Selected features ranked by importance:")
    for i, (feature, score) in enumerate(selected_scores, 1):
        print(f"   {i:2d}. {feature:<25} | Score: {score:.2f}")
    
    # For linear SVM, show coefficients
    if model.named_steps['svm'].kernel == 'linear':
        coefficients = model.named_steps['svm'].coef_[0]
        feature_importance = list(zip(selected_features, coefficients))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n⚖️  Linear SVM Feature Coefficients:")
        print("   (Positive = Spam indicator, Negative = Ham indicator)")
        for i, (feature, coef) in enumerate(feature_importance, 1):
            indicator = "📈 SPAM" if coef > 0 else "📉 HAM"
            print(f"   {i:2d}. {feature:<25} | {coef:+.4f} | {indicator}")
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        features_plot = [f[0] for f in feature_importance[:15]]
        coefs_plot = [f[1] for f in feature_importance[:15]]
        
        colors = ['red' if coef > 0 else 'blue' for coef in coefs_plot]
        bars = plt.barh(range(len(features_plot)), coefs_plot, color=colors, alpha=0.7)
        
        plt.yticks(range(len(features_plot)), features_plot)
        plt.xlabel('SVM Coefficient')
        plt.title('Feature Importance (Linear SVM Coefficients)\nRed = Spam Indicators, Blue = Ham Indicators', 
                 fontweight='bold', fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.8)
        
        # Add coefficient values
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + (0.001 if width > 0 else -0.001), bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left' if width > 0 else 'right', va='center', 
                    fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig("E:\\Baki\\svm_feature_importance.png", dpi=300, bbox_inches='tight')
        print(f"\n💾 Feature importance plot saved: E:\\Baki\\svm_feature_importance.png")
        plt.show()
    
    return selected_features

def test_email_samples(model, feature_columns):
    """
    Test the model with sample emails
    """
    print(f"\n🧪 TESTING MODEL WITH SAMPLE EMAILS:")
    print("="*60)
    
    # Note: Since we're using pre-computed features, we can't directly test raw text
    # Instead, we'll show how the model performs on the test set examples
    
    print("⚠️  Note: This model uses pre-computed features from CSV.")
    print("   To test new emails, you would need to:")
    print("   1. Extract the same features from the new email text")
    print("   2. Apply the same preprocessing pipeline")
    print("   3. Make predictions using the trained model")
    
    print(f"\n💡 Model is ready for deployment with feature extraction pipeline!")

# Main execution
if __name__ == "__main__":
    print("🚀 SVM SPAM DETECTION MODEL BUILDER")
    print("   (Using Pre-computed Features)")
    print("="*65)
    
    try:
        # Build and evaluate the model
        model, X_test, y_test, y_pred, y_pred_proba, features = build_svm_from_features_csv()
        
        # Test with samples
        test_email_samples(model, features)
        
        print(f"\n✅ SVM MODEL TRAINING COMPLETE!")
        print("="*60)
        print(f"📊 Final Performance Summary:")
        print(f"   • Accuracy:  {accuracy_score(y_test, y_pred):.1%}")
        print(f"   • F1-Score:  {f1_score(y_test, y_pred):.1%}")
        print(f"   • ROC AUC:   {roc_auc_score(y_test, y_pred_proba):.1%}")
        print(f"💾 Model saved: svm_spam_classifier_final.pkl")
        print(f"📈 Visualizations: svm_comprehensive_analysis.png")
        print(f"🎯 Ready for production deployment!")
        
    except FileNotFoundError:
        print("❌ Error: ml_features_matrix.csv not found!")
        print("💡 Please make sure the file exists at: E:\\Baki\\ml_features_matrix.csv")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Press Enter to exit...")
    try:
        input()
    except:
        pass