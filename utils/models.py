import gensim.downloader as api
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


#nltk.download('vader_lexicon')  # download necessary data for sentiment analysis


# Instantiate sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

glove_model = api.load('glove-twitter-25')

