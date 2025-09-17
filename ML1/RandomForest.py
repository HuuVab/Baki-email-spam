import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

def build_random_forest_from_csv(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """
    Build Random Forest spam classifier using pre-computed features CSV
    """
    print("🌲 RANDOM FOREST SPAM DETECTION MODEL")
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
    
    # Check for missing values and data quality
    missing_values = X.isnull().sum().sum()
    if missing_values > 0:
        print(f"⚠️  Found {missing_values} missing values. Filling with median...")
        X = X.fillna(X.median())
    else:
        print("✅ No missing values found")
    
    # Basic statistics
    print(f"\n📊 Feature statistics:")
    print(f"   • Features with zero variance: {(X.std() == 0).sum()}")
    print(f"   • Features with high correlation (>0.9): {check_high_correlations(X)}")
    
    # Split the data
    print(f"\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {len(X_train):,} emails")
    print(f"   Test set: {len(X_test):,} emails")
    
    # Analyze feature importance with correlation
    print(f"\n🔍 Initial feature correlation analysis...")
    feature_correlations = []
    for feature in feature_columns:
        corr = df[feature].corr(df['label'])
        feature_correlations.append((feature, abs(corr), corr))
    
    feature_correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Top 10 features by correlation with spam:")
    for i, (feature, abs_corr, corr) in enumerate(feature_correlations[:10], 1):
        direction = "↑ Spam" if corr > 0 else "↓ Ham"
        print(f"   {i:2d}. {feature:<25} | {abs_corr:.4f} | {direction}")
    
    # Create preprocessing pipeline (Random Forest doesn't need scaling but we'll include feature selection)
    print(f"\n⚙️ Creating preprocessing pipeline...")
    print("   Note: Random Forest doesn't require feature scaling")
    print("   But we'll include optional feature selection for comparison")
    
    # Create multiple pipeline configurations
    pipelines = {
        'rf_all_features': Pipeline([
            ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ]),
        'rf_selected_features': Pipeline([
            ('feature_selection', SelectKBest(f_classif)),
            ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
    }
    
    # Comprehensive hyperparameter tuning
    print(f"🎛️ Setting up hyperparameter search space...")
    
    # Parameters for Random Forest with all features
    param_grid_all = {
        'rf__n_estimators': [100, 200, 300, 500],
        'rf__max_depth': [10, 20, 30, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['sqrt', 'log2', None],
        'rf__bootstrap': [True, False],
        'rf__class_weight': [None, 'balanced']
    }
    
    # Parameters for Random Forest with feature selection
    param_grid_selected = {
        'feature_selection__k': [15, 20, 25, 'all'],
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [15, 25, None],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2],
        'rf__max_features': ['sqrt', 'log2'],
        'rf__class_weight': [None, 'balanced']
    }
    
    # Use RandomizedSearchCV for efficiency (Random Forest has many parameters)
    print(f"🔍 Performing Randomized Search (faster than Grid Search)...")
    print(f"   Testing both approaches: All features vs Selected features")
    
    best_models = {}
    
    # Search for best model with all features
    print(f"\n🌲 Training Random Forest with ALL features...")
    rs_all = RandomizedSearchCV(
        pipelines['rf_all_features'], 
        param_grid_all,
        n_iter=50,  # Try 50 random combinations
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    rs_all.fit(X_train, y_train)
    best_models['all_features'] = rs_all
    
    # Search for best model with feature selection
    print(f"\n🎯 Training Random Forest with SELECTED features...")
    rs_selected = RandomizedSearchCV(
        pipelines['rf_selected_features'],
        param_grid_selected,
        n_iter=30,  # Try 30 random combinations
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    rs_selected.fit(X_train, y_train)
    best_models['selected_features'] = rs_selected
    
    # Compare models and select the best
    print(f"\n🏆 COMPARING MODEL CONFIGURATIONS:")
    print("="*50)
    
    model_scores = {}
    for name, model in best_models.items():
        cv_score = model.best_score_
        model_scores[name] = cv_score
        print(f"   {name:<20} | CV F1 Score: {cv_score:.4f}")
    
    # Select best model
    best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k])
    best_rf = best_models[best_model_name].best_estimator_
    best_params = best_models[best_model_name].best_params_
    best_cv_score = best_models[best_model_name].best_score_
    
    print(f"\n✅ BEST MODEL SELECTED: {best_model_name.upper()}")
    print("="*50)
    print(f"Cross-validation F1 score: {best_cv_score:.4f}")
    print(f"Best parameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # Make predictions on test set
    print(f"\n🎯 Making predictions on test set...")
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
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
    
    # Classification report
    print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
    print("-" * 50)
    report = classification_report(y_test, y_pred, 
                                 target_names=['Ham (Legitimate)', 'Spam (Malicious)'],
                                 digits=4)
    print(report)
    
    # Cross-validation consistency
    print(f"\n🔄 CROSS-VALIDATION CONSISTENCY CHECK:")
    print("-" * 45)
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='f1')
    print(f"F1 Scores across 5 folds: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Standard deviation: {cv_scores.std():.4f}")
    
    # Analyze Random Forest specific insights
    analyze_rf_insights(best_rf, feature_columns, X_train, y_train)
    
    # Create comprehensive visualizations
    create_rf_visualizations(y_test, y_pred, y_pred_proba, best_rf, X_train, y_train, 
                            feature_columns, best_model_name)
    
    # Save the model
    model_path = "E:\\Baki\\random_forest_spam_classifier.pkl"
    joblib.dump(best_rf, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    return best_rf, X_test, y_test, y_pred, y_pred_proba, feature_columns

def check_high_correlations(X, threshold=0.9):
    """Check for highly correlated features"""
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(column, index, corr_matrix.loc[index, column]) 
                       for column in upper_triangle.columns 
                       for index in upper_triangle.index 
                       if upper_triangle.loc[index, column] > threshold]
    return len(high_corr_pairs)

def analyze_rf_insights(model, feature_columns, X_train, y_train):
    """
    Analyze Random Forest specific insights
    """
    print(f"\n🌲 RANDOM FOREST INSIGHTS:")
    print("="*50)
    
    # Get the Random Forest model (handle pipeline)
    if hasattr(model, 'named_steps'):
        if 'rf' in model.named_steps:
            rf_model = model.named_steps['rf']
            # Get feature names after selection if applicable
            if 'feature_selection' in model.named_steps:
                selector = model.named_steps['feature_selection']
                selected_mask = selector.get_support()
                used_features = [feature_columns[i] for i, selected in enumerate(selected_mask) if selected]
            else:
                used_features = feature_columns
        else:
            rf_model = model
            used_features = feature_columns
    else:
        rf_model = model
        used_features = feature_columns
    
    # Feature importance analysis
    feature_importance = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': used_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"🎯 Model Configuration:")
    print(f"   • Number of trees: {rf_model.n_estimators}")
    print(f"   • Max depth: {rf_model.max_depth}")
    print(f"   • Features used: {len(used_features)}")
    print(f"   • Min samples split: {rf_model.min_samples_split}")
    print(f"   • Min samples leaf: {rf_model.min_samples_leaf}")
    print(f"   • Max features: {rf_model.max_features}")
    
    print(f"\n🏆 TOP 15 MOST IMPORTANT FEATURES:")
    print("-" * 45)
    for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<25} | {row['importance']:.4f}")
    
    # Out-of-bag score if available
    if rf_model.bootstrap and hasattr(rf_model, 'oob_score_'):
        print(f"\n📊 Out-of-Bag Score: {rf_model.oob_score_:.4f}")
    
    # Feature importance distribution
    print(f"\n📈 Feature Importance Statistics:")
    print(f"   • Mean importance: {feature_importance.mean():.4f}")
    print(f"   • Std importance:  {feature_importance.std():.4f}")
    print(f"   • Max importance:  {feature_importance.max():.4f}")
    print(f"   • Min importance:  {feature_importance.min():.4f}")
    
    # Top features capture percentage
    top_5_importance = feature_importance_df.head(5)['importance'].sum()
    top_10_importance = feature_importance_df.head(10)['importance'].sum()
    
    print(f"\n🎯 Feature Concentration:")
    print(f"   • Top 5 features capture:  {top_5_importance:.1%} of total importance")
    print(f"   • Top 10 features capture: {top_10_importance:.1%} of total importance")
    
    return feature_importance_df

def create_rf_visualizations(y_test, y_pred, y_pred_proba, model, X_train, y_train, 
                            feature_columns, model_type):
    """
    Create comprehensive Random Forest visualization suite
    """
    print(f"\n🎨 Creating Random Forest visualizations...")
    
    # Set up the main plot
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle('Random Forest Spam Detection Model - Complete Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Get RF model and features
    if hasattr(model, 'named_steps'):
        if 'rf' in model.named_steps:
            rf_model = model.named_steps['rf']
            if 'feature_selection' in model.named_steps:
                selector = model.named_steps['feature_selection']
                selected_mask = selector.get_support()
                used_features = [feature_columns[i] for i, selected in enumerate(selected_mask) if selected]
            else:
                used_features = feature_columns
        else:
            rf_model = model
            used_features = feature_columns
    else:
        rf_model = model
        used_features = feature_columns
    
    # 1. Enhanced Confusion Matrix
    ax = axes[0, 0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax.set_title('Confusion Matrix\n(Random Forest)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Add percentages and error rates
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                   ha='center', va='center', fontsize=11, color='red')
    
    # 2. ROC Curve with Tree Ensemble Insight
    ax = axes[0, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    ax.plot(fpr, tpr, color='darkgreen', lw=3, label=f'RF ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkgreen')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve\n({rf_model.n_estimators} Trees)', fontweight='bold', fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    # 3. Feature Importance Plot
    ax = axes[0, 2]
    feature_importance = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': used_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(15)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 15 Feature Importance\n(Random Forest)', fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Add importance values
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    # 4. Tree Depth Distribution (if max_depth is not None)
    ax = axes[1, 0]
    if rf_model.max_depth is not None:
        tree_depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
        ax.hist(tree_depths, bins=20, color='forestgreen', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Tree Depth')
        ax.set_ylabel('Number of Trees')
        ax.set_title(f'Tree Depth Distribution\n(Mean: {np.mean(tree_depths):.1f})', 
                    fontweight='bold', fontsize=14)
        ax.grid(alpha=0.3)
    else:
        # Show feature usage instead
        feature_usage = np.zeros(len(used_features))
        for tree in rf_model.estimators_:
            feature_usage += (tree.feature_importances_ > 0).astype(int)
        
        top_used_features = pd.DataFrame({
            'feature': used_features,
            'usage_count': feature_usage
        }).sort_values('usage_count', ascending=False).head(15)
        
        bars = ax.barh(range(len(top_used_features)), top_used_features['usage_count'], 
                      color='forestgreen', alpha=0.7)
        ax.set_yticks(range(len(top_used_features)))
        ax.set_yticklabels(top_used_features['feature'], fontsize=9)
        ax.set_xlabel('Trees Using Feature')
        ax.set_title('Feature Usage Across Trees', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
    
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
    ax.set_title('RF Prediction Confidence\nDistribution', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 6. Performance Metrics Comparison
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
    ax.set_title('Performance Metrics\n(Random Forest)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Score')
    
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
    ax.set_title('Learning Curve\n(Ensemble Performance)', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 8. Feature Importance Cumulative Distribution
    ax = axes[2, 1]
    sorted_importance = np.sort(feature_importance)[::-1]
    cumulative_importance = np.cumsum(sorted_importance)
    
    ax.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
           'g-', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.8, label='80% Threshold')
    ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.8, label='90% Threshold')
    
    # Find features needed for 80% and 90% importance
    features_80 = np.argmax(cumulative_importance >= 0.8) + 1
    features_90 = np.argmax(cumulative_importance >= 0.9) + 1
    
    ax.axvline(x=features_80, color='red', linestyle=':', alpha=0.8)
    ax.axvline(x=features_90, color='orange', linestyle=':', alpha=0.8)
    
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cumulative Importance')
    ax.set_title(f'Feature Importance Distribution\n80%: {features_80} features, 90%: {features_90} features', 
                fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 9. Random Forest Model Summary
    ax = axes[2, 2]
    ax.axis('off')
    
    # Calculate additional metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    
    # Out-of-bag score
    oob_text = f"OOB Score: {rf_model.oob_score_:.4f}" if hasattr(rf_model, 'oob_score_') else "OOB: N/A"
    
    summary_text = f"""
RANDOM FOREST MODEL SUMMARY

Dataset: {len(y_test):,} test samples
Configuration:
- Trees: {rf_model.n_estimators}
- Max Depth: {rf_model.max_depth}
- Features Used: {len(used_features)}
- {oob_text}

PERFORMANCE METRICS:
- Accuracy:    {accuracy_score(y_test, y_pred):.4f}
- Precision:   {precision_score(y_test, y_pred):.4f}
- Recall:      {recall_score(y_test, y_pred):.4f}
- Specificity: {specificity:.4f}
- F1-Score:    {f1_score(y_test, y_pred):.4f}
- AUC-ROC:     {auc:.4f}

FEATURE INSIGHTS:
- Top feature: {importance_df.iloc[0]['feature']}
- Importance: {importance_df.iloc[0]['importance']:.4f}
- 80% captured by: {features_80} features

MODEL TYPE: {model_type.upper().replace('_', ' ')}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("E:\\Baki\\random_forest_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    print("💾 Random Forest analysis saved: E:\\Baki\\random_forest_comprehensive_analysis.png")
    plt.show()
    
    # Create separate feature importance plot
    create_detailed_feature_plot(rf_model, used_features)

def create_detailed_feature_plot(rf_model, feature_names):
    """Create detailed feature importance visualization"""
    
    print("🎯 Creating detailed feature importance plot...")
    
    # Get feature importance
    importance = rf_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Random Forest Feature Importance - Detailed Analysis', 
                 fontsize=18, fontweight='bold')
    
    # Left plot: Top 20 features
    ax1.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
    top_n = min(20, len(feature_names))
    
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    bars = ax1.barh(range(top_n), importance[indices[:top_n]], color=colors)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([feature_names[i] for i in indices[:top_n]])
    ax1.set_xlabel('Feature Importance')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add importance values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Right plot: Cumulative importance
    ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    cumsum_importance = np.cumsum(importance[indices])
    
    ax2.plot(range(1, len(cumsum_importance) + 1), cumsum_importance, 
             'b-', linewidth=3, marker='o', markersize=6)
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.8, linewidth=2, label='80%')
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='90%')
    ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.8, linewidth=2, label='95%')
    
    # Mark key points
    idx_80 = np.argmax(cumsum_importance >= 0.8) + 1
    idx_90 = np.argmax(cumsum_importance >= 0.9) + 1
    idx_95 = np.argmax(cumsum_importance >= 0.95) + 1
    
    ax2.scatter([idx_80, idx_90, idx_95], [0.8, 0.9, 0.95], 
               c=['red', 'orange', 'green'], s=100, zorder=5)
    
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Cumulative Importance')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Add annotations
    ax2.annotate(f'{idx_80} features', xy=(idx_80, 0.8), xytext=(idx_80+3, 0.75),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
    ax2.annotate(f'{idx_90} features', xy=(idx_90, 0.9), xytext=(idx_90+3, 0.85),
                arrowprops=dict(arrowstyle='->', color='orange'), fontsize=10, color='orange')
    
    plt.tight_layout()
    plt.savefig("E:\\Baki\\rf_feature_importance_detailed.png", dpi=300, bbox_inches='tight')
    print("💾 Detailed feature importance plot saved: E:\\Baki\\rf_feature_importance_detailed.png")
    plt.show()

def compare_models_performance(rf_model, X_test, y_test, feature_columns):
    """Compare Random Forest with other quick models"""
    
    print(f"\n⚖️ QUICK MODEL COMPARISON:")
    print("="*50)
    
    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    
    # Quick models for comparison
    models = {
        'Random Forest': rf_model,
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    }
    
    # Compare models
    results = {}
    for name, model in models.items():
        if name != 'Random Forest':  # RF is already trained
            print(f"   Training {name}...")
            model.fit(X_test.iloc[:5000], y_test.iloc[:5000])  # Quick training on subset
        
        y_pred = model.predict(X_test)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
    
    # Display comparison
    comparison_df = pd.DataFrame(results).T
    print(f"\n📊 Model Performance Comparison:")
    print(comparison_df.round(4))
    
    # Find best model
    best_model = comparison_df['f1'].idxmax()
    print(f"\n🏆 Best performing model: {best_model}")
    print(f"   F1-Score: {comparison_df.loc[best_model, 'f1']:.4f}")
    
    return comparison_df

def test_rf_on_samples(rf_model, feature_columns):
    """
    Demonstrate how to use the trained Random Forest model
    """
    print(f"\n🧪 RANDOM FOREST MODEL USAGE GUIDE:")
    print("="*60)
    
    print("📝 To use this model for new emails:")
    print("   1. Extract the same features from new email text")
    print("   2. Create a DataFrame with the feature columns")
    print("   3. Use rf_model.predict() and rf_model.predict_proba()")
    
    print(f"\n📋 Required feature columns ({len(feature_columns)}):")
    for i, feature in enumerate(feature_columns, 1):
        print(f"   {i:2d}. {feature}")
    
    print(f"\n💡 Model advantages:")
    print(f"   ✅ Handles missing values well")
    print(f"   ✅ No feature scaling required")
    print(f"   ✅ Provides feature importance")
    print(f"   ✅ Resistant to overfitting")
    print(f"   ✅ Fast prediction time")
    
    # Show model configuration
    if hasattr(rf_model, 'named_steps') and 'rf' in rf_model.named_steps:
        actual_rf = rf_model.named_steps['rf']
    else:
        actual_rf = rf_model
    
    print(f"\n⚙️ Model configuration:")
    print(f"   • Trees: {actual_rf.n_estimators}")
    print(f"   • Max depth: {actual_rf.max_depth}")
    print(f"   • Min samples split: {actual_rf.min_samples_split}")
    print(f"   • Min samples leaf: {actual_rf.min_samples_leaf}")
    print(f"   • Bootstrap: {actual_rf.bootstrap}")

# Main execution
if __name__ == "__main__":
    print("🌲 RANDOM FOREST SPAM DETECTION MODEL BUILDER")
    print("="*70)
    
    try:
        # Build and evaluate the Random Forest model
        rf_model, X_test, y_test, y_pred, y_pred_proba, features = build_random_forest_from_csv()
        
        # Compare with other models
        comparison_results = compare_models_performance(rf_model, X_test, y_test, features)
        
        # Usage guide
        test_rf_on_samples(rf_model, features)
        
        print(f"\n✅ RANDOM FOREST MODEL TRAINING COMPLETE!")
        print("="*65)
        print(f"🎯 Final Performance Summary:")
        print(f"   • Accuracy:  {accuracy_score(y_test, y_pred):.1%}")
        print(f"   • F1-Score:  {f1_score(y_test, y_pred):.1%}")
        print(f"   • Precision: {precision_score(y_test, y_pred):.1%}")
        print(f"   • Recall:    {recall_score(y_test, y_pred):.1%}")
        print(f"   • ROC AUC:   {roc_auc_score(y_test, y_pred_proba):.1%}")
        
        print(f"\n📁 Generated Files:")
        print(f"   💾 Model: random_forest_spam_classifier.pkl")
        print(f"   📊 Analysis: random_forest_comprehensive_analysis.png")
        print(f"   🎯 Features: rf_feature_importance_detailed.png")
        
        print(f"\n🌲 Random Forest Advantages:")
        print(f"   ✅ Excellent performance out-of-the-box")
        print(f"   ✅ Built-in feature importance ranking")
        print(f"   ✅ Handles overfitting well with multiple trees")
        print(f"   ✅ No preprocessing required (scales well)")
        print(f"   ✅ Fast training and prediction")
        print(f"   ✅ Robust to outliers and missing values")
        
        # Feature insights
        if hasattr(rf_model, 'named_steps') and 'rf' in rf_model.named_steps:
            actual_rf = rf_model.named_steps['rf']
        else:
            actual_rf = rf_model
            
        top_feature_idx = np.argmax(actual_rf.feature_importances_)
        print(f"\n🎯 Key Insights:")
        print(f"   🏆 Most important feature: {features[top_feature_idx]}")
        print(f"   📈 Feature importance: {actual_rf.feature_importances_[top_feature_idx]:.4f}")
        print(f"   🌲 Trees in ensemble: {actual_rf.n_estimators}")
        
        print(f"\n🚀 Model ready for production deployment!")
        
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