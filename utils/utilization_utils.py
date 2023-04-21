# required imports
from transformers import pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import numpy as np

from . import models

analyzer = models.analyzer


def get_sentiment(text):
    """Function to apply sentiment analysis to text"""
    return analyzer.polarity_scores(text)['compound']


def get_vader_sentiment_for_df(df, column_names:list): 
    """Apply Vader sentiment analysis to DataFrame and create new column with results"""
    df = df.copy()
    for col in column_names:
        df[col + '_vader_sentiment_pred'] = df[col].apply(get_sentiment)  # add a colum to the df
    return df


def get_hf_sentiment_for_df(df, column_names:list):
    """Apply Hugging Face sentiment analysis to DataFrame and create new column with results"""
    df = df.copy()
    for col in column_names:
        df[col + '_hf_sentiment_pred'] = df[col].apply(hugging_sentiment)  # add a colum to the df
    return df


def hugging_sentiment(text):
    """sentiment analysis on a single document - return only sentiment score. 
    Positive number - positive sentiment.
    Negative number - negative sentiment.
    Size - porbability."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    # if we want to insert a corpus and not a single text:
    #  chance "text" with "data" where: data = df['txt'].tolist()
    score = sentiment_pipeline(text)[0]['score']
    if sentiment_pipeline(text)[0]['label'] =='NEGATIVE':
        score = score*(-1)
    return score  #  0.999708354473114


def sentiment_test(df, txt_col, label_col='sentiment'):
    """ Runs sentiment predition and compares it to the label"""
    # split to train test
    train, test = train_test_split(df, test_size=0.2)  # split row's wise

    train_x = train[txt_col]
    test_x = test[txt_col]

    # Fitting data
    count_vect = CountVectorizer()
    count_vect.fit(train_x)

    # Transforming data
    train_texts_vec = count_vect.transform(train_x)
    test_texts_vec = count_vect.transform(test_x)

    # define a model
    nb = MultinomialNB()
    # train a model
    nb.fit(train_texts_vec, train[label_col])
    # predict
    y_pred = nb.predict(test_texts_vec)

    # return score
    #acc = accuracy_score(test[label_col], y_pred)
    cv_acc = np.mean(cross_val_score(nb, test_texts_vec, test[label_col], cv=8))

    return cv_acc





