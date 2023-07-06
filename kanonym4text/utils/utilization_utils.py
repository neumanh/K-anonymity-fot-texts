# required imports
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import logging

def get_mean_semantice_distance_for_corpus(cor1, cor2, plot=False):
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
    if plot:
        plot_hist(data=cosine_dist, xlabel='Semantic Distance', fig_name=None)
    
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
    if fig_name:
        plt.savefig(fig_name)
    else:
        plt.show()


def embedded_dist(embed1, embed2):
    """
    this function recieve 2 embedings and return the cosine dist between the two
    we will use it to show the dist between embedings of a sentence before and after anonymization
    """

    # compute cosine similarity
    cosine = np.dot(embed1, embed2) / (norm(embed1) * norm(embed2))
    embed_dist = 1 - cosine
    return embed_dist


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



