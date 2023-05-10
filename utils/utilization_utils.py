# required imports
# from transformers import pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


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



