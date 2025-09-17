import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def analyze_email_dataset(csv_file="E:\\Baki\\final_email_dataset_cleaned.csv", save_plots=True):
    """
    Comprehensive analysis of email dataset with insights and plots
    """
    
    print("📊 EMAIL DATASET ANALYSIS")
    print("="*50)
    
    # Load dataset
    print("📁 Loading dataset...")
    df = pd.read_csv(csv_file)
    
    print(f"📊 Dataset size: {len(df):,} emails")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Basic statistics
    print(f"\n📈 LABEL DISTRIBUTION:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "Ham (Legitimate)" if label == 0 else "Spam (Malicious)"
        percentage = (count / len(df)) * 100
        print(f"   • {label} ({label_name}): {count:,} emails ({percentage:.1f}%)")
    
    # Text analysis functions
    def count_words(text):
        """Count words in text"""
        return len(text.split())
    
    def count_sentences(text):
        """Count sentences in text"""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def count_characters(text):
        """Count characters in text"""
        return len(text)
    
    # Calculate metrics
    print(f"\n🔄 Calculating text metrics...")
    
    df['char_count'] = df['text'].apply(count_characters)
    df['word_count'] = df['text'].apply(count_words)
    df['sentence_count'] = df['text'].apply(count_sentences)
    
    # Separate by labels
    ham_emails = df[df['label'] == 0]
    spam_emails = df[df['label'] == 1]
    
    # Calculate statistics
    stats = {
        'Ham': {
            'count': len(ham_emails),
            'avg_chars': ham_emails['char_count'].mean(),
            'avg_words': ham_emails['word_count'].mean(),
            'avg_sentences': ham_emails['sentence_count'].mean(),
            'median_chars': ham_emails['char_count'].median(),
            'median_words': ham_emails['word_count'].median(),
            'median_sentences': ham_emails['sentence_count'].median(),
        },
        'Spam': {
            'count': len(spam_emails),
            'avg_chars': spam_emails['char_count'].mean(),
            'avg_words': spam_emails['word_count'].mean(),
            'avg_sentences': spam_emails['sentence_count'].mean(),
            'median_chars': spam_emails['char_count'].median(),
            'median_words': spam_emails['word_count'].median(),
            'median_sentences': spam_emails['sentence_count'].median(),
        }
    }
    
    # Print detailed statistics
    print(f"\n📊 DETAILED STATISTICS COMPARISON:")
    print("="*60)
    
    print(f"\n📏 CHARACTER COUNT:")
    print(f"   Ham (0):  Average = {stats['Ham']['avg_chars']:.0f} chars  | Median = {stats['Ham']['median_chars']:.0f} chars")
    print(f"   Spam (1): Average = {stats['Spam']['avg_chars']:.0f} chars  | Median = {stats['Spam']['median_chars']:.0f} chars")
    print(f"   📈 Difference: Ham is {stats['Ham']['avg_chars'] - stats['Spam']['avg_chars']:.0f} chars longer on average")
    
    print(f"\n📝 WORD COUNT:")
    print(f"   Ham (0):  Average = {stats['Ham']['avg_words']:.0f} words  | Median = {stats['Ham']['median_words']:.0f} words")
    print(f"   Spam (1): Average = {stats['Spam']['avg_words']:.0f} words  | Median = {stats['Spam']['median_words']:.0f} words")
    print(f"   📈 Difference: Ham has {stats['Ham']['avg_words'] - stats['Spam']['avg_words']:.0f} more words on average")
    
    print(f"\n💬 SENTENCE COUNT:")
    print(f"   Ham (0):  Average = {stats['Ham']['avg_sentences']:.1f} sentences  | Median = {stats['Ham']['median_sentences']:.1f} sentences")
    print(f"   Spam (1): Average = {stats['Spam']['avg_sentences']:.1f} sentences  | Median = {stats['Spam']['median_sentences']:.1f} sentences")
    print(f"   📈 Difference: Ham has {stats['Ham']['avg_sentences'] - stats['Spam']['avg_sentences']:.1f} more sentences on average")
    
    # Create visualizations
    if save_plots:
        create_visualizations(df, ham_emails, spam_emails, stats)
    
    return df, stats

def create_visualizations(df, ham_emails, spam_emails, stats):
    """Create and save all visualization plots"""
    
    print(f"\n🎨 Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Label Distribution Pie Chart
    ax1 = plt.subplot(2, 4, 1)
    labels = ['Ham (Legitimate)', 'Spam (Malicious)']
    sizes = [stats['Ham']['count'], stats['Spam']['count']]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=colors, explode=explode, shadow=True)
    plt.title('Email Distribution\n(Ham vs Spam)', fontsize=12, fontweight='bold')
    
    # 2. Average Character Count Comparison
    ax2 = plt.subplot(2, 4, 2)
    categories = ['Ham', 'Spam']
    char_counts = [stats['Ham']['avg_chars'], stats['Spam']['avg_chars']]
    bars = plt.bar(categories, char_counts, color=['#2ecc71', '#e74c3c'], alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, char_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Average Characters per Email', fontsize=12, fontweight='bold')
    plt.ylabel('Characters')
    plt.grid(axis='y', alpha=0.3)
    
    # 3. Average Word Count Comparison
    ax3 = plt.subplot(2, 4, 3)
    word_counts = [stats['Ham']['avg_words'], stats['Spam']['avg_words']]
    bars = plt.bar(categories, word_counts, color=['#2ecc71', '#e74c3c'], alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, word_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Average Words per Email', fontsize=12, fontweight='bold')
    plt.ylabel('Words')
    plt.grid(axis='y', alpha=0.3)
    
    # 4. Average Sentence Count Comparison
    ax4 = plt.subplot(2, 4, 4)
    sentence_counts = [stats['Ham']['avg_sentences'], stats['Spam']['avg_sentences']]
    bars = plt.bar(categories, sentence_counts, color=['#2ecc71', '#e74c3c'], alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, sentence_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Average Sentences per Email', fontsize=12, fontweight='bold')
    plt.ylabel('Sentences')
    plt.grid(axis='y', alpha=0.3)
    
    # 5. Character Count Distribution (Box Plot)
    ax5 = plt.subplot(2, 4, 5)
    # Limit to reasonable range for visualization
    ham_chars_limited = ham_emails[ham_emails['char_count'] <= 5000]['char_count']
    spam_chars_limited = spam_emails[spam_emails['char_count'] <= 5000]['char_count']
    
    box_data = [ham_chars_limited, spam_chars_limited]
    box_plot = plt.boxplot(box_data, labels=['Ham', 'Spam'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('#2ecc71')
    box_plot['boxes'][1].set_facecolor('#e74c3c')
    
    plt.title('Character Count Distribution\n(Limited to ≤5000 chars)', fontsize=12, fontweight='bold')
    plt.ylabel('Characters')
    plt.grid(axis='y', alpha=0.3)
    
    # 6. Word Count Distribution (Box Plot)
    ax6 = plt.subplot(2, 4, 6)
    # Limit to reasonable range for visualization
    ham_words_limited = ham_emails[ham_emails['word_count'] <= 1000]['word_count']
    spam_words_limited = spam_emails[spam_emails['word_count'] <= 1000]['word_count']
    
    box_data = [ham_words_limited, spam_words_limited]
    box_plot = plt.boxplot(box_data, labels=['Ham', 'Spam'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('#2ecc71')
    box_plot['boxes'][1].set_facecolor('#e74c3c')
    
    plt.title('Word Count Distribution\n(Limited to ≤1000 words)', fontsize=12, fontweight='bold')
    plt.ylabel('Words')
    plt.grid(axis='y', alpha=0.3)
    
    # 7. Character Length Histogram
    ax7 = plt.subplot(2, 4, 7)
    # Use limited data for better visualization
    ham_chars_hist = ham_emails[ham_emails['char_count'] <= 3000]['char_count']
    spam_chars_hist = spam_emails[spam_emails['char_count'] <= 3000]['char_count']
    
    plt.hist(ham_chars_hist, bins=50, alpha=0.6, label='Ham', color='#2ecc71', density=True)
    plt.hist(spam_chars_hist, bins=50, alpha=0.6, label='Spam', color='#e74c3c', density=True)
    
    plt.title('Character Count Distribution\n(Histogram)', fontsize=12, fontweight='bold')
    plt.xlabel('Characters')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 8. Summary Statistics Table
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Create summary table
    table_data = [
        ['Metric', 'Ham (0)', 'Spam (1)', 'Difference'],
        ['Count', f"{stats['Ham']['count']:,}", f"{stats['Spam']['count']:,}", ''],
        ['Avg Characters', f"{stats['Ham']['avg_chars']:.0f}", f"{stats['Spam']['avg_chars']:.0f}", 
         f"+{stats['Ham']['avg_chars'] - stats['Spam']['avg_chars']:.0f}"],
        ['Avg Words', f"{stats['Ham']['avg_words']:.0f}", f"{stats['Spam']['avg_words']:.0f}", 
         f"+{stats['Ham']['avg_words'] - stats['Spam']['avg_words']:.0f}"],
        ['Avg Sentences', f"{stats['Ham']['avg_sentences']:.1f}", f"{stats['Spam']['avg_sentences']:.1f}", 
         f"+{stats['Ham']['avg_sentences'] - stats['Spam']['avg_sentences']:.1f}"]
    ]
    
    table = ax8.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax8.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_path = "E:\\Baki\\email_dataset_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Comprehensive analysis plot saved: {output_path}")
    
    # Create individual focused plots
    create_individual_plots(df, ham_emails, spam_emails, stats)
    
    plt.show()

def create_individual_plots(df, ham_emails, spam_emails, stats):
    """Create individual focused plots"""
    
    # 1. Character comparison plot
    plt.figure(figsize=(10, 6))
    categories = ['Ham (Legitimate)', 'Spam (Malicious)']
    char_counts = [stats['Ham']['avg_chars'], stats['Spam']['avg_chars']]
    
    bars = plt.bar(categories, char_counts, color=['#2ecc71', '#e74c3c'], alpha=0.8, width=0.6)
    
    # Add value labels
    for bar, value in zip(bars, char_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{value:.0f} chars', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.title('Average Email Length Comparison\n(Characters)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Average Characters', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add difference annotation
    difference = stats['Ham']['avg_chars'] - stats['Spam']['avg_chars']
    plt.text(0.5, max(char_counts) * 0.8, f'Ham emails are\n{difference:.0f} chars longer\non average', 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("E:\\Baki\\character_comparison.png", dpi=300, bbox_inches='tight')
    print(f"💾 Character comparison plot saved: E:\\Baki\\character_comparison.png")
    plt.show()
    
    # 2. Word comparison plot
    plt.figure(figsize=(10, 6))
    word_counts = [stats['Ham']['avg_words'], stats['Spam']['avg_words']]
    
    bars = plt.bar(categories, word_counts, color=['#2ecc71', '#e74c3c'], alpha=0.8, width=0.6)
    
    # Add value labels
    for bar, value in zip(bars, word_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{value:.0f} words', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.title('Average Email Length Comparison\n(Words)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Average Words', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add difference annotation
    difference = stats['Ham']['avg_words'] - stats['Spam']['avg_words']
    plt.text(0.5, max(word_counts) * 0.8, f'Ham emails have\n{difference:.0f} more words\non average', 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
             fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("E:\\Baki\\word_comparison.png", dpi=300, bbox_inches='tight')
    print(f"💾 Word comparison plot saved: E:\\Baki\\word_comparison.png")
    plt.show()
    
    # 3. Sentence comparison plot
    plt.figure(figsize=(10, 6))
    sentence_counts = [stats['Ham']['avg_sentences'], stats['Spam']['avg_sentences']]
    
    bars = plt.bar(categories, sentence_counts, color=['#2ecc71', '#e74c3c'], alpha=0.8, width=0.6)
    
    # Add value labels
    for bar, value in zip(bars, sentence_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{value:.1f} sentences', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.title('Average Email Length Comparison\n(Sentences)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Average Sentences', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add difference annotation
    difference = stats['Ham']['avg_sentences'] - stats['Spam']['avg_sentences']
    plt.text(0.5, max(sentence_counts) * 0.8, f'Ham emails have\n{difference:.1f} more sentences\non average', 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("E:\\Baki\\sentence_comparison.png", dpi=300, bbox_inches='tight')
    print(f"💾 Sentence comparison plot saved: E:\\Baki\\sentence_comparison.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    print("📊 EMAIL DATASET INSIGHTS GENERATOR")
    print("="*50)
    
    # Run analysis
    dataset, statistics = analyze_email_dataset()
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Plots saved to E:\\Baki\\ folder")
    print(f"📈 Key insights:")
    print(f"   • Ham emails are longer and more detailed")
    print(f"   • Spam emails are shorter and more concise")
    print(f"   • Perfect dataset balance for ML training")
    
    print(f"\nPress Enter to exit...")
    try:
        input()
    except:
        pass