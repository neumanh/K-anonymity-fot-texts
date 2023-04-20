import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import pdist, squareform

import nlp_utils

# CountVectorizer is defined only once
vectorizer = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                           stop_words='english')

def get_bow(corpus, create_df = False):
    """ Vectorizes the corpus using CountVectorizer """

    cc = nlp_utils.clean_corpus(corpus)

    count_data = vectorizer.fit_transform(cc)

    if create_df:
        #create dataframe
        bow_dataframe = pd.DataFrame(count_data.toarray(),columns=vectorizer.get_feature_names_out())
    else:
        bow_dataframe = None
    return count_data, bow_dataframe


def get_anonym_degree(docs, min_k = None):
    """ If K not given, returns the minimal current k and the corresponding documents.
        If k is given, return the documents with k or less neighbohrs  """
    
    # Lemmatizing the documents
    ldocs = nlp_utils.clean_corpus(docs)

    # Vectorizing
    count_data = vectorizer.fit_transform(ldocs)
    
    # Counting unique values
    uniq_arr, uniq_cnt = np.unique(count_data.toarray(), axis=0, return_counts=True)
    if not min_k:
        min_k = min(uniq_cnt)
    
    # All the unique vectors
    un_anon = uniq_arr[uniq_cnt <= min_k]

    # Getting the unique vectore indeces
    indeces_list = []
    for row in un_anon:
        # Get the similar rows
        similar_vals = np.where((count_data.toarray() == (row)).all(axis=1))
        indeces_list.append(similar_vals[0].tolist())

    return min_k, indeces_list


def get_dist_matrix(sparse_mat, metric='Jaccard'):
    """
    Calculates the distance matrix
    """
    # Create distance matrixe
    dist_cond = pdist(sparse_mat.todense(), metric)
    # Convert a vector-form distance vector to a square-form distance matrix
    dist_mtrx = squareform(dist_cond)
    # To prevent self-similarity
    np.fill_diagonal(dist_mtrx, np.inf)

def create_word_list(doc):
    # Remove stopwords and lemmatize
    doc = nlp_utils.clean_doc(doc)
    # Remove duplicates
    words = list(set(doc.split(' ')))
    return words

def get_diff_and_common(doc1, doc2):
    """
    Returns the common and non-common words between two lists
    """
    l1 = create_word_list(doc1)
    l2 = create_word_list(doc2)
    diff = list(set(l1) - set(l2))
    comm = list(set(l1).intersection(l2))
    return diff, comm


if __name__ == 'main':
    print('Hi')

    diff, comm = get_diff_and_common('hi i am hadas and i love banana', 'hi he is john and he loves hummus')