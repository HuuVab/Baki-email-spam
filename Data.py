import pandas as pd

def show_examples_from_dataset(csv_file="E:\\Baki\\final_email_dataset.csv"):
    """
    Show examples of ham (0) and spam (1) emails from the final dataset
    """
    
    print("🔍 LOADING DATASET AND SHOWING EXAMPLES")
    print("="*70)
    
    # Load the dataset
    df = pd.read_csv(csv_file)
    
    print(f"📊 Dataset loaded: {len(df):,} rows")
    print(f"📋 Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Show HAM examples (label = 0)
    print(f"\n📩 HAM EMAILS (Label = 0) - Legitimate Emails:")
    print("="*70)
    
    ham_emails = df[df['label'] == 0]['text'].head(10)
    
    for i, email in enumerate(ham_emails, 1):
        print(f"\n🟢 HAM Example {i}:")
        print("-" * 40)
        # Show first 300 characters for readability
        email_preview = email[:300] + "..." if len(email) > 300 else email
        print(f"{email_preview}")
    
    # Show SPAM examples (label = 1)
    print(f"\n\n🚨 SPAM EMAILS (Label = 1) - Malicious/Spam Emails:")
    print("="*70)
    
    spam_emails = df[df['label'] == 1]['text'].head(10)
    
    for i, email in enumerate(spam_emails, 1):
        print(f"\n🔴 SPAM Example {i}:")
        print("-" * 40)
        # Show first 300 characters for readability
        email_preview = email[:300] + "..." if len(email) > 300 else email
        print(f"{email_preview}")
    
    # Show some statistics
    print(f"\n\n📊 TEXT LENGTH STATISTICS:")
    print("="*40)
    
    ham_lengths = df[df['label'] == 0]['text'].str.len()
    spam_lengths = df[df['label'] == 1]['text'].str.len()
    
    print(f"HAM emails (label=0):")
    print(f"  • Average length: {ham_lengths.mean():.0f} characters")
    print(f"  • Min length: {ham_lengths.min()} characters")
    print(f"  • Max length: {ham_lengths.max():,} characters")
    
    print(f"\nSPAM emails (label=1):")
    print(f"  • Average length: {spam_lengths.mean():.0f} characters")
    print(f"  • Min length: {spam_lengths.min()} characters")
    print(f"  • Max length: {spam_lengths.max():,} characters")

# Run the example viewer
if __name__ == "__main__":
    show_examples_from_dataset()
    
    print(f"\n👋 Press Enter to exit...")
    try:
        input()
    except:
        pass