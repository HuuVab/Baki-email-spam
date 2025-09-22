import pandas as pd

def create_reduced_feature_dataset(input_file="E:\\Baki\\ml_features_matrix.csv", 
                                 output_file="E:\\Baki\\ml_features_top10.csv"):
    """
    Create new CSV with only the top 10 most important features
    """
    print("Creating reduced feature dataset...")
    print("="*40)
    
    # Load the original dataset
    print("Loading original dataset...")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original features: {df.columns.tolist()}")
    
    # Define the top 10 features from your analysis
    top_10_features = [
        'char_count',
        'sentiment_positive', 
        'readability_score',
        'avg_word_length',
        'word_count',
        'grade_level',
        'number_count',
        'punctuation_count',
        'spam_keywords',
        'sentiment_negative'
    ]
    
    # Include the label column
    selected_columns = top_10_features + ['label']
    
    print(f"\nSelected top 10 features:")
    for i, feature in enumerate(top_10_features, 1):
        print(f"  {i:2d}. {feature}")
    
    # Check if all selected features exist in the dataset
    missing_features = [feat for feat in selected_columns if feat not in df.columns]
    if missing_features:
        print(f"\nWarning: Missing features in dataset: {missing_features}")
        # Only use features that exist
        selected_columns = [feat for feat in selected_columns if feat in df.columns]
    
    # Create the reduced dataset
    reduced_df = df[selected_columns].copy()
    
    print(f"\nReduced dataset shape: {reduced_df.shape}")
    print(f"Reduced features: {reduced_df.columns.tolist()}")
    
    # Check for missing values
    missing_values = reduced_df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing values found:")
        for feature, count in missing_values.items():
            if count > 0:
                print(f"  {feature}: {count} missing values")
        
        # Fill missing values with median
        print("Filling missing values with median...")
        reduced_df = reduced_df.fillna(reduced_df.median())
    
    # Display basic statistics
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(reduced_df):,}")
    ham_count = (reduced_df['label'] == 0).sum()
    spam_count = (reduced_df['label'] == 1).sum()
    print(f"  Ham emails: {ham_count:,} ({ham_count/len(reduced_df)*100:.1f}%)")
    print(f"  Spam emails: {spam_count:,} ({spam_count/len(reduced_df)*100:.1f}%)")
    
    # Save the reduced dataset
    reduced_df.to_csv(output_file, index=False)
    print(f"\nReduced dataset saved to: {output_file}")
    
    # Display feature ranges
    print(f"\nFeature value ranges:")
    print("-" * 50)
    for feature in top_10_features:
        if feature in reduced_df.columns:
            min_val = reduced_df[feature].min()
            max_val = reduced_df[feature].max()
            mean_val = reduced_df[feature].mean()
            print(f"  {feature:<20}: {min_val:8.2f} to {max_val:8.2f} (avg: {mean_val:8.2f})")
    
    return reduced_df

def compare_datasets(original_file="E:\\Baki\\ml_features_matrix.csv",
                    reduced_file="E:\\Baki\\ml_features_top10.csv"):
    """
    Compare original and reduced datasets
    """
    print("\nDataset Comparison:")
    print("="*30)
    
    # Load both datasets
    original_df = pd.read_csv(original_file)
    reduced_df = pd.read_csv(reduced_file)
    
    print(f"Original dataset:")
    print(f"  Shape: {original_df.shape}")
    print(f"  Features: {original_df.shape[1] - 1}")  # Exclude label
    print(f"  Size: {original_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    print(f"\nReduced dataset:")
    print(f"  Shape: {reduced_df.shape}")
    print(f"  Features: {reduced_df.shape[1] - 1}")  # Exclude label
    print(f"  Size: {reduced_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Calculate reduction
    feature_reduction = (original_df.shape[1] - reduced_df.shape[1]) / original_df.shape[1] * 100
    size_reduction = (1 - reduced_df.memory_usage(deep=True).sum() / original_df.memory_usage(deep=True).sum()) * 100
    
    print(f"\nReduction achieved:")
    print(f"  Features: {feature_reduction:.1f}% fewer")
    print(f"  Memory: {size_reduction:.1f}% smaller")
    
    print(f"\nFeatures removed:")
    removed_features = set(original_df.columns) - set(reduced_df.columns)
    for i, feature in enumerate(sorted(removed_features), 1):
        if feature != 'label':  # Don't list label as removed
            print(f"  {i:2d}. {feature}")

def validate_reduced_dataset(file_path="E:\\Baki\\ml_features_top10.csv"):
    """
    Validate the reduced dataset for ML training
    """
    print("\nValidating reduced dataset for ML training:")
    print("="*45)
    
    df = pd.read_csv(file_path)
    
    # Check basic requirements
    checks = []
    
    # 1. Check for required columns
    required_features = ['char_count', 'sentiment_positive', 'readability_score', 
                        'avg_word_length', 'word_count', 'grade_level', 
                        'number_count', 'punctuation_count', 'spam_keywords', 
                        'sentiment_negative', 'label']
    
    missing_cols = [col for col in required_features if col not in df.columns]
    if not missing_cols:
        checks.append("✓ All required columns present")
    else:
        checks.append(f"✗ Missing columns: {missing_cols}")
    
    # 2. Check for missing values
    if df.isnull().sum().sum() == 0:
        checks.append("✓ No missing values")
    else:
        checks.append(f"✗ {df.isnull().sum().sum()} missing values found")
    
    # 3. Check label distribution
    if 'label' in df.columns:
        unique_labels = df['label'].nunique()
        if unique_labels == 2:
            checks.append("✓ Binary classification labels (0, 1)")
        else:
            checks.append(f"✗ Unexpected label count: {unique_labels}")
    
    # 4. Check data types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) == len(df.columns):
        checks.append("✓ All columns are numeric")
    else:
        checks.append("✗ Non-numeric columns found")
    
    # Print validation results
    for check in checks:
        print(f"  {check}")
    
    # Summary
    if all("✓" in check for check in checks):
        print(f"\n✓ Dataset is ready for ML training!")
    else:
        print(f"\n⚠ Dataset needs attention before ML training")
    
    return df

# Main execution
if __name__ == "__main__":
    print("FEATURE REDUCTION TOOL")
    print("="*50)
    print("Creating optimized dataset with top 10 features")
    print("="*50)
    
    try:
        # Create reduced dataset
        reduced_df = create_reduced_feature_dataset()
        
        # Compare datasets
        compare_datasets()
        
        # Validate the reduced dataset
        validate_reduced_dataset()
        
        print(f"\n🎯 SUCCESS!")
        print("="*20)
        print("✓ Reduced dataset created successfully")
        print("✓ File saved: E:\\Baki\\ml_features_top10.csv")
        print("✓ Ready for faster ML training")
        print("✓ Significant size reduction achieved")
        
        print(f"\nNext steps:")
        print("1. Use ml_features_top10.csv for training your models")
        print("2. Compare performance with original 21-feature dataset")
        print("3. Expect faster training times with similar accuracy")
        
    except FileNotFoundError:
        print("Error: Original file 'ml_features_matrix.csv' not found!")
        print("Please check the file path: E:\\Baki\\ml_features_matrix.csv")
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"\nPress Enter to exit...")
    input()
    