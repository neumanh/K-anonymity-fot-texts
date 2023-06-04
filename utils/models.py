import gensim.downloader as api
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# nltk.download('vader_lexicon')  # download necessary data for sentiment analysis
# nltk.download('stopwords')  # TEMP


# Instantiate sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# model_name = 'glove-twitter-25'
model_name = 'word2vec-google-news-300'

glove_model = api.load(model_name)

