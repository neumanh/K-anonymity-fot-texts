# required imports
from transformers import pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import logging

from . import models

analyzer = models.analyzer

def get_mean_semantice_distance_for_corpus(cor1, cor2, prefix=None):
    """
    Calculates the distance for each pair of documents
    """
    sem_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1')
    cor1_embed = sem_model.encode(cor1, show_progress_bar=False)
    cor2_embed = sem_model.encode(cor2, show_progress_bar=False)

    if len(cor1) != len(cor2):
        logging.error('The two copuses must be in the same size.')
        return
    
    # dist_list = embedded_dist(cor1_embed, cor2_embed)
    # dist_list = cosine(cor1_embed, cor2_embed)
    # print('dist_list:', dist_list.shape)
    # # compute cosine similarity
    cosine_sim = np.sum(cor1_embed * cor2_embed, axis=1)/(norm(cor1_embed, axis=1)*norm(cor2_embed, axis=1))
    cosine_dist = 1 - cosine_sim
    
    # print('cosine_dist:', cosine_dist.shape)
    # dist_list = []
    #
    # for doc1, doc2 in zip(cor1, cor2):
    #     dist_list.append(get_semantice_distance_for_docs(doc1, doc2, sem_model))
    mean_dist = np.mean(cosine_dist)

    # Plotting a histogram
    if prefix:
        plot_hist(data=cosine_dist, xlabel='Semantic Distance', fig_name=f'plots/{prefix}_semantic_hist.pdf')
    
    return mean_dist


def get_semantice_distance_for_docs(doc1, doc2, sem_model):
    """
    Generates sentence embedding and calculates the distance between them
    """
    embed1 = sem_model.encode(doc1, show_progress_bar=False)
    embed2 = sem_model.encode(doc2, show_progress_bar=False)
    dist = embedded_dist(embed1, embed2)
    return dist


def plot_hist(data, xlabel, fig_name):
    """
    Plots a histogram and saves it.
    """
    plt.hist(data, bins=30)
    plt.ylabel('Count')
    plt.xlabel(xlabel)
    plt.savefig(fig_name)


def embedded_dist(embed1, embed2):
    """
    this function recieve 2 embedings and return the cosine dist between the two
    we will use it to show the dist between embedings of a sentence before and after anonymization
    """

    # compute cosine similarity
    cosine = np.dot(embed1, embed2) / (norm(embed1) * norm(embed2))
    embed_dist = 1 - cosine
    return embed_dist


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
    # train, test = train_test_split(df, test_size=0.2)  # split row's wise

    # train_x = train[txt_col]
    # test_x = test[txt_col]
    # print(train_x)
    #print(train[label_col])

    # Fitting data
    count_vect = CountVectorizer()
    # count_vect.fit(train_x)

    # # Transforming data
    # train_texts_vec = count_vect.transform(train_x)
    # test_texts_vec = count_vect.transform(test_x)

     # define a model
    nb = MultinomialNB()
    # # train a model
    # nb.fit(train_texts_vec, train[label_col])
    # # predict
    # y_pred = nb.predict(test_texts_vec)

    # return score
    #acc = accuracy_score(test[label_col], y_pred)
    # cv_acc = np.mean(cross_val_score(nb, test_texts_vec, test[label_col], cv=8))

    X = count_vect.fit_transform(df[txt_col])
    y = df[label_col]
    cv_acc = np.mean(cross_val_score(nb, X, y, cv=8))

    return cv_acc


def sentiment_test_xgb(df, txt_col, label_col='sentiment'):
    """
    Runs sentiment analysis using XGboost
    """
    # Use CountVectorizer to convert the text data into numerical features
    vectorizer = CountVectorizer(stop_words="english")

    # Train an XGBoost model on the training set
    model = xgb.XGBClassifier()

    X = vectorizer.fit_transform(df[txt_col])
    y = df[label_col]
    scores = cross_val_score(model, X, y, cv=8)

    return np.mean(scores)


def plot_sentiment_scatter(x, y):
    """
    Plots a scatter plot with colors. 
    blue dot - same sentiment
    red dot - different sentiment
    """
    # Same sentiment
    x_t = x[x*y > 0]
    y_t = y[x*y > 0]

    # Different sentiment
    x_f = x[x*y <= 0]
    y_f = y[x*y <= 0]

    plt.scatter(x_t, y_t, color='royalblue', label='Same sentiment')
    plt.scatter(x_f, y_f, color='firebrick', label='Invert sentiment')

    # plt.scatter(df_21['txt_vader_sentiment_pred'], df_21['anon_txt_vader_sentiment_pred'])
    plt.xlabel('Before annoymization')
    plt.ylabel('After annonymization')
    plt.legend()
    plt.title('Sentiment score before and after annonymization')
    plt.show()



