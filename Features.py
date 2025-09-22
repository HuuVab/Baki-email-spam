import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns

def analyze_feature_importance(csv_file="E:\\Baki\\ml_features_matrix.csv"):
    """Comprehensive feature importance analysis"""
    
    # Load data
    df = pd.read_csv(csv_file)
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"Analyzing {len(feature_columns)} features...")
    
    # 1. Statistical Tests
    print("\n1. STATISTICAL FEATURE SELECTION:")
    print("-" * 40)
    
    # Chi-square test (for non-negative features)
    X_positive = np.abs(X)  # Make positive for chi-square
    chi2_scores, chi2_pvalues = chi2(X_positive, y)
    
    # F-test
    f_scores, f_pvalues = f_classif(X, y)
    
    # Mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create feature analysis dataframe
    feature_analysis = pd.DataFrame({
        'feature': feature_columns,
        'chi2_score': chi2_scores,
        'f_score': f_scores,
        'mutual_info': mi_scores,
        'chi2_pvalue': chi2_pvalues,
        'f_pvalue': f_pvalues
    })
    
    # Rank features by each method
    feature_analysis['chi2_rank'] = feature_analysis['chi2_score'].rank(ascending=False)
    feature_analysis['f_rank'] = feature_analysis['f_score'].rank(ascending=False)
    feature_analysis['mi_rank'] = feature_analysis['mutual_info'].rank(ascending=False)
    
    # Average rank
    feature_analysis['avg_rank'] = (feature_analysis['chi2_rank'] + 
                                   feature_analysis['f_rank'] + 
                                   feature_analysis['mi_rank']) / 3
    
    # Sort by average rank
    feature_analysis = feature_analysis.sort_values('avg_rank')
    
    print("Top 10 features by statistical tests:")
    for i, row in feature_analysis.head(10).iterrows():
        print(f"{int(row['avg_rank']):2d}. {row['feature']:<25} | Avg Rank: {row['avg_rank']:.1f}")
    
    # 2. Model-based importance
    print("\n2. MODEL-BASED FEATURE IMPORTANCE:")
    print("-" * 40)
    
    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    # Logistic Regression coefficients
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)
    lr_importance = np.abs(lr.coef_[0])
    
    # Add to analysis
    feature_analysis['rf_importance'] = rf_importance
    feature_analysis['lr_importance'] = lr_importance
    feature_analysis['rf_rank_model'] = pd.Series(rf_importance).rank(ascending=False).values
    feature_analysis['lr_rank_model'] = pd.Series(lr_importance).rank(ascending=False).values
    
    # Combined ranking including models
    feature_analysis['final_rank'] = (feature_analysis['chi2_rank'] + 
                                     feature_analysis['f_rank'] + 
                                     feature_analysis['mi_rank'] +
                                     feature_analysis['rf_rank_model'] +
                                     feature_analysis['lr_rank_model']) / 5
    
    feature_analysis = feature_analysis.sort_values('final_rank')
    
    print("Top 10 features by combined analysis:")
    for i, row in feature_analysis.head(10).iterrows():
        print(f"{int(row['final_rank']):2d}. {row['feature']:<25} | Final Rank: {row['final_rank']:.1f}")
    
    # 3. Correlation analysis
    print("\n3. FEATURE CORRELATION WITH TARGET:")
    print("-" * 40)
    
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print("Top 10 by correlation with spam label:")
    for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"{i:2d}. {feature:<25} | Correlation: {corr:.4f}")
    
    # 4. Class difference analysis
    print("\n4. CLASS DIFFERENCE ANALYSIS:")
    print("-" * 40)
    
    ham_data = X[y == 0]
    spam_data = X[y == 1]
    
    class_differences = []
    for feature in feature_columns:
        ham_mean = ham_data[feature].mean()
        spam_mean = spam_data[feature].mean()
        difference = abs(spam_mean - ham_mean)
        relative_diff = difference / (abs(ham_mean) + abs(spam_mean) + 1e-8)  # Avoid division by zero
        class_differences.append((feature, difference, relative_diff, ham_mean, spam_mean))
    
    class_differences.sort(key=lambda x: x[2], reverse=True)  # Sort by relative difference
    
    print("Top 10 by class difference (Ham vs Spam):")
    for i, (feature, abs_diff, rel_diff, ham_mean, spam_mean) in enumerate(class_differences[:10], 1):
        print(f"{i:2d}. {feature:<25} | Ham: {ham_mean:8.2f} | Spam: {spam_mean:8.2f} | Diff: {rel_diff:.3f}")
    
    # 5. Create visualization
    create_feature_importance_plot(feature_analysis, correlations, class_differences)
    
    # 6. Feature selection recommendations
    print("\n5. FEATURE SELECTION RECOMMENDATIONS:")
    print("-" * 45)
    
    top_features = feature_analysis.head(10)['feature'].tolist()
    print(f"RECOMMENDED TOP 10 FEATURES:")
    for i, feature in enumerate(top_features, 1):
        print(f"{i:2d}. {feature}")
    
    # Save analysis
    feature_analysis.to_csv("E:\\Baki\\feature_importance_analysis.csv", index=False)
    print(f"\nDetailed analysis saved to: E:\\Baki\\feature_importance_analysis.csv")
    
    return feature_analysis, top_features

def create_feature_importance_plot(feature_analysis, correlations, class_differences):
    """Create comprehensive feature importance visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Importance Analysis - All Methods', fontsize=16, fontweight='bold')
    
    # 1. Combined ranking
    top_features = feature_analysis.head(10)
    ax1 = axes[0, 0]
    bars = ax1.barh(range(len(top_features)), top_features['final_rank'], color='skyblue')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels([f.replace('_', ' ').title() for f in top_features['feature']])
    ax1.set_xlabel('Average Rank (lower is better)')
    ax1.set_title('Top 10 Features - Combined Ranking')
    ax1.invert_yaxis()
    
    # 2. Correlation with target
    top_corr = correlations.head(10)
    ax2 = axes[0, 1]
    bars = ax2.barh(range(len(top_corr)), top_corr.values, color='lightgreen')
    ax2.set_yticks(range(len(top_corr)))
    ax2.set_yticklabels([f.replace('_', ' ').title() for f in top_corr.index])
    ax2.set_xlabel('Absolute Correlation')
    ax2.set_title('Top 10 Features - Correlation with Target')
    ax2.invert_yaxis()
    
    # 3. Random Forest importance
    rf_top = feature_analysis.nlargest(10, 'rf_importance')
    ax3 = axes[1, 0]
    bars = ax3.barh(range(len(rf_top)), rf_top['rf_importance'], color='orange')
    ax3.set_yticks(range(len(rf_top)))
    ax3.set_yticklabels([f.replace('_', ' ').title() for f in rf_top['feature']])
    ax3.set_xlabel('Random Forest Importance')
    ax3.set_title('Top 10 Features - Random Forest')
    ax3.invert_yaxis()
    
    # 4. Class differences
    class_diff_features = [x[0] for x in class_differences[:10]]
    class_diff_values = [x[2] for x in class_differences[:10]]
    ax4 = axes[1, 1]
    bars = ax4.barh(range(len(class_diff_features)), class_diff_values, color='pink')
    ax4.set_yticks(range(len(class_diff_features)))
    ax4.set_yticklabels([f.replace('_', ' ').title() for f in class_diff_features])
    ax4.set_xlabel('Relative Difference (Ham vs Spam)')
    ax4.set_title('Top 10 Features - Class Difference')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("E:\\Baki\\feature_importance_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

# Run the analysis
if __name__ == "__main__":
    feature_analysis, recommended_features = analyze_feature_importance()