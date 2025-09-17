import nltk
try:
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
except:
    pass