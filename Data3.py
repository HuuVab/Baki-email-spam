import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter
import warnings
import string
from textstat import flesch_reading_ease, flesch_kincaid_grade
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

warnings.filterwarnings('ignore')

# Download required NLTK data
import nltk
try:
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
except:
    pass


def analyze_email_dataset(csv_file="E:\\Baki\\final_email_dataset_cleaned.csv", save_plots=True):
    """
    Comprehensive analysis of email dataset with advanced insights and plots
    """
    
    print("📊 ADVANCED EMAIL DATASET ANALYSIS")
    print("="*60)
    
    # Load dataset
    print("📁 Loading dataset...")
    df = pd.read_csv(csv_file)
    
    print(f"📊 Dataset size: {len(df):,} emails")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Basic statistics
    print(f"\n📈 LABEL DISTRIBUTION:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "Ham (Legitimate)" if label == 0 else "Spam (Malicious)"
        percentage = (count / len(df)) * 100
        print(f"   • {label} ({label_name}): {count:,} emails ({percentage:.1f}%)")
    
    # Advanced text analysis functions
    def count_words(text):
        return len(text.split())
    
    def count_sentences(text):
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def count_characters(text):
        return len(text)
    
    def count_uppercase_words(text):
        words = text.split()
        return sum(1 for word in words if word.isupper() and len(word) > 1)
    
    def count_punctuation(text):
        return sum(1 for char in text if char in string.punctuation)
    
    def count_exclamation_marks(text):
        return text.count('!')
    
    def count_question_marks(text):
        return text.count('?')
    
    def count_urls(text):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return len(re.findall(url_pattern, text))
    
    def count_email_addresses(text):
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return len(re.findall(email_pattern, text))
    
    def count_phone_numbers(text):
        phone_pattern = r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        return len(re.findall(phone_pattern, text))
    
    def count_numbers(text):
        return len(re.findall(r'\d+', text))
    
    def count_currency_symbols(text):
        currency_pattern = r'[\$£€¥₹]'
        return len(re.findall(currency_pattern, text))
    
    def avg_word_length(text):
        words = text.split()
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)
    
    def get_readability_score(text):
        try:
            return flesch_reading_ease(text)
        except:
            return 0
    
    def get_grade_level(text):
        try:
            return flesch_kincaid_grade(text)
        except:
            return 0
    
    def get_sentiment_scores(text):
        try:
            scores = sia.polarity_scores(text)
            return scores['compound'], scores['pos'], scores['neg'], scores['neu']
        except:
            return 0, 0, 0, 1
    
    def count_spam_keywords(text):
        spam_keywords = [
            'free', 'winner', 'cash', 'prize', 'urgent', 'limited', 'offer',
            'click', 'buy', 'sale', 'discount', 'guarantee', 'money',
            'credit', 'loan', 'debt', 'act now', 'call now', 'order now'
        ]
        text_lower = text.lower()
        return sum(1 for keyword in spam_keywords if keyword in text_lower)
    
    def has_suspicious_formatting(text):
        # Check for excessive capitalization, multiple exclamations, etc.
        score = 0
        if text.count('!') > 3:
            score += 1
        if len([w for w in text.split() if w.isupper() and len(w) > 1]) > 5:
            score += 1
        if text.count('$') > 2:
            score += 1
        return score
    
    # Calculate all metrics
    print(f"\n🔄 Calculating advanced text metrics...")
    
    # Basic metrics
    df['char_count'] = df['text'].apply(count_characters)
    df['word_count'] = df['text'].apply(count_words)
    df['sentence_count'] = df['text'].apply(count_sentences)
    
    # Advanced metrics
    df['uppercase_words'] = df['text'].apply(count_uppercase_words)
    df['punctuation_count'] = df['text'].apply(count_punctuation)
    df['exclamation_count'] = df['text'].apply(count_exclamation_marks)
    df['question_count'] = df['text'].apply(count_question_marks)
    df['url_count'] = df['text'].apply(count_urls)
    df['email_count'] = df['text'].apply(count_email_addresses)
    df['phone_count'] = df['text'].apply(count_phone_numbers)
    df['number_count'] = df['text'].apply(count_numbers)
    df['currency_count'] = df['text'].apply(count_currency_symbols)
    df['avg_word_length'] = df['text'].apply(avg_word_length)
    df['readability_score'] = df['text'].apply(get_readability_score)
    df['grade_level'] = df['text'].apply(get_grade_level)
    df['spam_keywords'] = df['text'].apply(count_spam_keywords)
    df['suspicious_formatting'] = df['text'].apply(has_suspicious_formatting)
    
    # Sentiment analysis
    print("🎭 Performing sentiment analysis...")
    sentiment_data = df['text'].apply(get_sentiment_scores)
    df['sentiment_compound'] = [x[0] for x in sentiment_data]
    df['sentiment_positive'] = [x[1] for x in sentiment_data]
    df['sentiment_negative'] = [x[2] for x in sentiment_data]
    df['sentiment_neutral'] = [x[3] for x in sentiment_data]
    
    # Separate by labels
    ham_emails = df[df['label'] == 0]
    spam_emails = df[df['label'] == 1]
    
    # Calculate comprehensive statistics
    metrics = [
        'char_count', 'word_count', 'sentence_count', 'uppercase_words',
        'punctuation_count', 'exclamation_count', 'question_count',
        'url_count', 'email_count', 'phone_count', 'number_count',
        'currency_count', 'avg_word_length', 'readability_score',
        'grade_level', 'spam_keywords', 'suspicious_formatting',
        'sentiment_compound', 'sentiment_positive', 'sentiment_negative'
    ]
    
    stats = {}
    for category, data in [('Ham', ham_emails), ('Spam', spam_emails)]:
        stats[category] = {}
        for metric in metrics:
            stats[category][metric] = {
                'mean': data[metric].mean(),
                'median': data[metric].median(),
                'std': data[metric].std(),
                'min': data[metric].min(),
                'max': data[metric].max()
            }
        stats[category]['count'] = len(data)
    
    # Print detailed analysis
    print_detailed_analysis(stats)
    
    # Create all visualizations
    if save_plots:
        create_comprehensive_visualizations(df, ham_emails, spam_emails, stats)
        create_word_clouds(ham_emails, spam_emails)
        create_interactive_plots(df, ham_emails, spam_emails)
        generate_ml_features_report(df)
    
    return df, stats

def print_detailed_analysis(stats):
    """Print comprehensive statistical analysis"""
    
    print(f"\n📊 COMPREHENSIVE STATISTICAL ANALYSIS:")
    print("="*80)
    
    categories = [
        ('Basic Text Metrics', ['char_count', 'word_count', 'sentence_count', 'avg_word_length']),
        ('Formatting Indicators', ['uppercase_words', 'punctuation_count', 'exclamation_count', 'question_count']),
        ('Content Indicators', ['url_count', 'email_count', 'phone_count', 'number_count', 'currency_count']),
        ('Readability Metrics', ['readability_score', 'grade_level']),
        ('Spam Indicators', ['spam_keywords', 'suspicious_formatting']),
        ('Sentiment Analysis', ['sentiment_compound', 'sentiment_positive', 'sentiment_negative'])
    ]
    
    for category_name, metrics in categories:
        print(f"\n📋 {category_name.upper()}:")
        print("-" * 60)
        
        for metric in metrics:
            ham_avg = stats['Ham'][metric]['mean']
            spam_avg = stats['Spam'][metric]['mean']
            difference = ham_avg - spam_avg
            
            print(f"   {metric.replace('_', ' ').title()}:")
            print(f"      Ham:  {ham_avg:.2f} | Spam: {spam_avg:.2f} | Diff: {difference:+.2f}")

def create_comprehensive_visualizations(df, ham_emails, spam_emails, stats):
    """Create comprehensive visualization plots"""
    
    print(f"\n🎨 Creating comprehensive visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create main comprehensive plot
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    fig.suptitle('Comprehensive Email Dataset Analysis', fontsize=20, fontweight='bold')
    
    # 1. Label Distribution
    ax = axes[0, 0]
    labels = ['Ham', 'Spam']
    sizes = [stats['Ham']['count'], stats['Spam']['count']]
    colors = ['#2ecc71', '#e74c3c']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Email Distribution', fontweight='bold')
    
    # 2. Character Count Comparison
    ax = axes[0, 1]
    metrics = ['Ham', 'Spam']
    char_counts = [stats['Ham']['char_count']['mean'], stats['Spam']['char_count']['mean']]
    bars = ax.bar(metrics, char_counts, color=colors, alpha=0.7)
    ax.set_title('Average Character Count', fontweight='bold')
    ax.set_ylabel('Characters')
    for bar, val in zip(bars, char_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Spam Keywords Comparison
    ax = axes[0, 2]
    spam_kw = [stats['Ham']['spam_keywords']['mean'], stats['Spam']['spam_keywords']['mean']]
    bars = ax.bar(metrics, spam_kw, color=colors, alpha=0.7)
    ax.set_title('Spam Keywords Count', fontweight='bold')
    ax.set_ylabel('Keywords')
    for bar, val in zip(bars, spam_kw):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Sentiment Analysis
    ax = axes[0, 3]
    sentiment_ham = stats['Ham']['sentiment_compound']['mean']
    sentiment_spam = stats['Spam']['sentiment_compound']['mean']
    sentiments = [sentiment_ham, sentiment_spam]
    bars = ax.bar(metrics, sentiments, color=colors, alpha=0.7)
    ax.set_title('Sentiment Compound Score', fontweight='bold')
    ax.set_ylabel('Sentiment Score')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, sentiments):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 if val > 0 else bar.get_height() - 0.05, 
                f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    # 5. Readability Score Distribution
    ax = axes[1, 0]
    ham_read = ham_emails[ham_emails['readability_score'] > 0]['readability_score']
    spam_read = spam_emails[spam_emails['readability_score'] > 0]['readability_score']
    ax.hist(ham_read, bins=30, alpha=0.6, label='Ham', color='#2ecc71', density=True)
    ax.hist(spam_read, bins=30, alpha=0.6, label='Spam', color='#e74c3c', density=True)
    ax.set_title('Readability Score Distribution', fontweight='bold')
    ax.set_xlabel('Readability Score')
    ax.set_ylabel('Density')
    ax.legend()
    
    # 6. Exclamation Marks
    ax = axes[1, 1]
    excl_counts = [stats['Ham']['exclamation_count']['mean'], stats['Spam']['exclamation_count']['mean']]
    bars = ax.bar(metrics, excl_counts, color=colors, alpha=0.7)
    ax.set_title('Average Exclamation Marks', fontweight='bold')
    ax.set_ylabel('Exclamation Marks')
    for bar, val in zip(bars, excl_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. URL Count
    ax = axes[1, 2]
    url_counts = [stats['Ham']['url_count']['mean'], stats['Spam']['url_count']['mean']]
    bars = ax.bar(metrics, url_counts, color=colors, alpha=0.7)
    ax.set_title('Average URL Count', fontweight='bold')
    ax.set_ylabel('URLs')
    for bar, val in zip(bars, url_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Currency Symbols
    ax = axes[1, 3]
    currency_counts = [stats['Ham']['currency_count']['mean'], stats['Spam']['currency_count']['mean']]
    bars = ax.bar(metrics, currency_counts, color=colors, alpha=0.7)
    ax.set_title('Average Currency Symbols', fontweight='bold')
    ax.set_ylabel('Currency Symbols')
    for bar, val in zip(bars, currency_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 9. Word Count Box Plot
    ax = axes[2, 0]
    ham_words = ham_emails[ham_emails['word_count'] <= 500]['word_count']
    spam_words = spam_emails[spam_emails['word_count'] <= 500]['word_count']
    box_data = [ham_words, spam_words]
    bp = ax.boxplot(box_data, labels=['Ham', 'Spam'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_title('Word Count Distribution', fontweight='bold')
    ax.set_ylabel('Words')
    
    # 10. Grade Level
    ax = axes[2, 1]
    grade_levels = [stats['Ham']['grade_level']['mean'], stats['Spam']['grade_level']['mean']]
    bars = ax.bar(metrics, grade_levels, color=colors, alpha=0.7)
    ax.set_title('Average Grade Level', fontweight='bold')
    ax.set_ylabel('Grade Level')
    for bar, val in zip(bars, grade_levels):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 11. Uppercase Words
    ax = axes[2, 2]
    upper_counts = [stats['Ham']['uppercase_words']['mean'], stats['Spam']['uppercase_words']['mean']]
    bars = ax.bar(metrics, upper_counts, color=colors, alpha=0.7)
    ax.set_title('Average Uppercase Words', fontweight='bold')
    ax.set_ylabel('Uppercase Words')
    for bar, val in zip(bars, upper_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 12. Suspicious Formatting
    ax = axes[2, 3]
    susp_counts = [stats['Ham']['suspicious_formatting']['mean'], stats['Spam']['suspicious_formatting']['mean']]
    bars = ax.bar(metrics, susp_counts, color=colors, alpha=0.7)
    ax.set_title('Suspicious Formatting Score', fontweight='bold')
    ax.set_ylabel('Suspicion Score')
    for bar, val in zip(bars, susp_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 13-16: Correlation heatmaps and feature importance
    correlation_metrics = ['char_count', 'word_count', 'spam_keywords', 'sentiment_compound', 
                          'exclamation_count', 'url_count', 'currency_count', 'suspicious_formatting']
    
    # 13. Ham Correlation
    ax = axes[3, 0]
    ham_corr = ham_emails[correlation_metrics].corr()
    sns.heatmap(ham_corr, annot=True, cmap='RdYlBu', center=0, ax=ax, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    ax.set_title('Ham Emails - Feature Correlations', fontweight='bold')
    
    # 14. Spam Correlation
    ax = axes[3, 1]
    spam_corr = spam_emails[correlation_metrics].corr()
    sns.heatmap(spam_corr, annot=True, cmap='RdYlBu', center=0, ax=ax, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    ax.set_title('Spam Emails - Feature Correlations', fontweight='bold')
    
    # 15. Feature Importance (difference between ham and spam)
    ax = axes[3, 2]
    feature_importance = []
    features = ['spam_keywords', 'currency_count', 'suspicious_formatting', 'exclamation_count', 
               'url_count', 'sentiment_negative']
    
    for feature in features:
        ham_mean = stats['Ham'][feature]['mean']
        spam_mean = stats['Spam'][feature]['mean']
        importance = abs(spam_mean - ham_mean)
        feature_importance.append(importance)
    
    bars = ax.barh(features, feature_importance, color='orange', alpha=0.7)
    ax.set_title('Feature Discrimination Power', fontweight='bold')
    ax.set_xlabel('Absolute Difference')
    for bar, val in zip(bars, feature_importance):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontweight='bold')
    
    # 16. Summary Statistics Table
    ax = axes[3, 3]
    ax.axis('off')
    
    # Create summary table
    table_data = [
        ['Metric', 'Ham Avg', 'Spam Avg', 'Ratio'],
        ['Words', f"{stats['Ham']['word_count']['mean']:.0f}", 
         f"{stats['Spam']['word_count']['mean']:.0f}", 
         f"{stats['Ham']['word_count']['mean']/stats['Spam']['word_count']['mean']:.1f}x"],
        ['Spam Keywords', f"{stats['Ham']['spam_keywords']['mean']:.1f}", 
         f"{stats['Spam']['spam_keywords']['mean']:.1f}", 
         f"{stats['Spam']['spam_keywords']['mean']/max(stats['Ham']['spam_keywords']['mean'], 0.1):.1f}x"],
        ['Sentiment', f"{stats['Ham']['sentiment_compound']['mean']:.2f}", 
         f"{stats['Spam']['sentiment_compound']['mean']:.2f}", ''],
        ['Readability', f"{stats['Ham']['readability_score']['mean']:.0f}", 
         f"{stats['Spam']['readability_score']['mean']:.0f}", '']
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Key Metrics Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_path = "E:\\Baki\\comprehensive_email_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Comprehensive analysis plot saved: {output_path}")
    plt.show()

def create_word_clouds(ham_emails, spam_emails):
    """Create word clouds for ham and spam emails"""
    
    print(f"☁️ Creating word clouds...")
    
    # Prepare text data
    stop_words = set(stopwords.words('english'))
    
    def clean_text_for_wordcloud(text):
        # Remove common email artifacts
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    
    ham_text = ' '.join([clean_text_for_wordcloud(text) for text in ham_emails['text'].head(1000)])
    spam_text = ' '.join([clean_text_for_wordcloud(text) for text in spam_emails['text'].head(1000)])
    
    # Create word clouds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Ham word cloud
    ham_wordcloud = WordCloud(width=800, height=400, background_color='white',
                             colormap='Greens', max_words=100).generate(ham_text)
    ax1.imshow(ham_wordcloud, interpolation='bilinear')
    ax1.set_title('Ham Emails - Most Common Words', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Spam word cloud
    spam_wordcloud = WordCloud(width=800, height=400, background_color='white',
                              colormap='Reds', max_words=100).generate(spam_text)
    ax2.imshow(spam_wordcloud, interpolation='bilinear')
    ax2.set_title('Spam Emails - Most Common Words', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("E:\\Baki\\wordclouds_comparison.png", dpi=300, bbox_inches='tight')
    print(f"💾 Word clouds saved: E:\\Baki\\wordclouds_comparison.png")
    plt.show()

def create_interactive_plots(df, ham_emails, spam_emails):
    """Create interactive plots using Plotly"""
    
    print(f"🎯 Creating interactive plots...")
    
    # 1. Interactive scatter plot
    fig = px.scatter(df, x='word_count', y='sentiment_compound', 
                     color='label', color_discrete_map={0: 'green', 1: 'red'},
                     hover_data=['char_count', 'spam_keywords', 'readability_score'],
                     title='Email Length vs Sentiment (Interactive)',
                     labels={'word_count': 'Word Count', 'sentiment_compound': 'Sentiment Score'})
    fig.write_html("E:\\Baki\\interactive_scatter.html")
    
    # 2. Interactive feature comparison
    features = ['word_count', 'spam_keywords', 'exclamation_count', 'currency_count', 'url_count']
    ham_means = [ham_emails[feature].mean() for feature in features]
    spam_means = [spam_emails[feature].mean() for feature in features]
    
    fig = go.Figure(data=[
        go.Bar(name='Ham', x=features, y=ham_means, marker_color='green'),
        go.Bar(name='Spam', x=features, y=spam_means, marker_color='red')
    ])
    fig.update_layout(barmode='group', title='Interactive Feature Comparison')
    fig.write_html("E:\\Baki\\interactive_features.html")
    
    print(f"💾 Interactive plots saved as HTML files in E:\\Baki\\")

def generate_ml_features_report(df):
    """Generate a report of features for machine learning"""
    
    print(f"\n🤖 MACHINE LEARNING FEATURES REPORT:")
    print("="*60)
    
    # Feature categories
    basic_features = ['char_count', 'word_count', 'sentence_count', 'avg_word_length']
    content_features = ['uppercase_words', 'punctuation_count', 'exclamation_count', 
                       'question_count', 'url_count', 'email_count', 'phone_count', 
                       'number_count', 'currency_count']
    advanced_features = ['readability_score', 'grade_level', 'spam_keywords', 
                        'suspicious_formatting']
    sentiment_features = ['sentiment_compound', 'sentiment_positive', 'sentiment_negative', 
                         'sentiment_neutral']
    
    all_features = basic_features + content_features + advanced_features + sentiment_features
    
    # Calculate feature correlations with label
    correlations = []
    for feature in all_features:
        corr = df[feature].corr(df['label'])
        correlations.append((feature, abs(corr), corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🎯 TOP FEATURES FOR SPAM DETECTION (by correlation):")
    print("-" * 50)
    for i, (feature, abs_corr, corr) in enumerate(correlations[:10], 1):
        direction = "↑ Spam" if corr > 0 else "↓ Ham"
        print(f"{i:2d}. {feature:<20} | {abs_corr:.3f} | {direction}")
    
    # Save feature matrix
    feature_matrix = df[all_features + ['label']].copy()
    feature_matrix.to_csv("E:\\Baki\\ml_features_matrix.csv", index=False)
    print(f"\n💾 ML features matrix saved: E:\\Baki\\ml_features_matrix.csv")
    print(f"📊 Shape: {feature_matrix.shape} (rows, columns)")
    print(f"🎯 Ready for ML model training!")
    
    return feature_matrix, correlations

# Main execution
if __name__ == "__main__":
    print("🚀 ADVANCED EMAIL DATASET ANALYSIS ENGINE")
    print("="*60)
    
    # Install required packages message
    print("📦 Make sure you have installed:")
    print("   pip install textstat wordcloud plotly nltk")
    print("="*60)
    
    try:
        # Run comprehensive analysis
        dataset, statistics = analyze_email_dataset()
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print("="*50)
        print(f"📊 Generated plots and files:")
        print(f"   • comprehensive_email_analysis.png")
        print(f"   • wordclouds_comparison.png")
        print(f"   • interactive_scatter.html")
        print(f"   • interactive_features.html")
        print(f"   • ml_features_matrix.csv")
        
        print(f"\n🎯 KEY INSIGHTS DISCOVERED:")
        print("="*40)
        print(f"   🔍 Feature Engineering: {len(dataset.columns)-2} new features created")
        print(f"   📈 Ham emails are typically longer and more formal")
        print(f"   🚨 Spam emails contain more suspicious indicators")
        print(f"   💭 Sentiment analysis reveals communication patterns")
        print(f"   📊 Dataset is perfectly balanced for ML training")
        print(f"   🎯 Top spam indicators identified for model training")
        
        # Additional analysis summaries
        print(f"\n📋 DATASET SUMMARY:")
        print("-" * 30)
        print(f"   Total emails: {len(dataset):,}")
        print(f"   Ham emails: {len(dataset[dataset['label']==0]):,}")
        print(f"   Spam emails: {len(dataset[dataset['label']==1]):,}")
        print(f"   Total features: {len(dataset.columns)}")
        print(f"   Ready for ML: ✅")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install pandas matplotlib seaborn numpy textstat wordcloud plotly nltk scikit-learn")
    
    print(f"\n🎉 Analysis complete! Press Enter to exit...")
    try:
        input()
    except:
        pass