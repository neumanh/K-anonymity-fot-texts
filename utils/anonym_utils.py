import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import pdist, squareform
import re

from . import nlp_utils

# CountVectorizer is defined only once
vectorizer = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                           stop_words='english')

def get_bow(corpus):
    """ Vectorizes the corpus using CountVectorizer """

    cc = nlp_utils.clean_corpus(corpus)

    count_data = vectorizer.fit_transform(cc)
    voc = vectorizer.get_feature_names_out()

    # if create_df:
    #     #create dataframe
    #     bow_dataframe = pd.DataFrame(count_data.toarray(),columns=vectorizer.get_feature_names_out())
    # else:
    #     bow_dataframe = None
    return count_data, voc


def get_anonym_degree(docs = None, vecs = None, min_k = None):
    """ If K not given, returns the minimal current k and the corresponding documents.
        If k is given, return the documents with k or less neighbohrs  """
    
    if docs is not None:
        # Lemmatizing the documents
        ldocs = nlp_utils.clean_corpus(docs)

        # Vectorizing
        count_data = vectorizer.fit_transform(ldocs)
    elif vecs is not None:
        count_data = vecs
    else:
        print('You must supply documents or vectors')
        return
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


def get_diff(vecs):
    """
    Finds the differences between arrays of 0,1.
    The input is a sparse matrix
    """
    # Creating a matrix
    mat = np.vstack(vecs)
    #nparray = sparse_mat.toarray()
    xsum = np.sum(mat, axis=0)

    union_x = (xsum > 0).astype('int')
    inter_x = (xsum == len(vecs)).astype('int')

    diff = union_x - inter_x
    return diff


def jaccard_index(u, v):
    """
    For vectors of 0 and 1
    Creadit: JDWarner https://gist.github.com/JDWarner/6730886
    """
    if np.double(np.bitwise_or(u, v).sum()) != 0:
        j = np.double(np.bitwise_and(u, v).sum()) / np.double(np.bitwise_or(u, v).sum())
    else:
        j = 0

    return j


def get_nearest_neighbors(n_list, k):
    """
    Finds the k nearest neighbors for each item.
    returns list of neighbor list (list of lists)
    """
    # Chossing the jaccard index method (for string of vectors)
    if isinstance(n_list[0], str):
        jaccard_func = nlp_utils.jaccard_index
    else:
        jaccard_func = jaccard_index
    
    all_neighbors = []
    used_items = []
    for idx1, doc1 in enumerate(n_list):
        if idx1 not in used_items:
            #  Appending to the used items
            used_items.append(idx1)
            doc1_neighbors = []
            for idx2, doc2 in enumerate(n_list):
                    if idx2 not in used_items:
                        # Appending a tuple of (distance, index)
                        doc1_neighbors.append((jaccard_func(doc1, doc2), idx2))
                        #print(idx1, idx2, jaccard_func(doc1, doc2))

            # Sorting accroding to the first item in the tuple - distance
            doc1_neighbors.sort(reverse=True)
            # Looking for k-1 neighbors
            doc1_neighbors = doc1_neighbors[:k-1] 
            # Adding the document itself, so 
            #if doc1 not in 
            curr_doc_list = [idx1]
            for curr_doc in doc1_neighbors:
                # Adding the neighbor
                curr_doc_list.append(curr_doc[1])
                # Removing from the availble document list
                used_items.append(curr_doc[1])
            all_neighbors.append(curr_doc_list)
            #all_neighbors.append(free_items)
    return all_neighbors


def force_anonym(df, k, col='anon_txt'):
    """
    Force anonymity by:
    1. Finding nearest neighbors
    2. Finding the different words
    2. Replacing the different words to *
    """
    df = df.copy()
    vecs, voc = get_bow(df[col])
    curr_k, non_anon_indexes = get_anonym_degree(docs=df[col])
    print('get_anonym_degree:', curr_k)
    # Flattening the list of lists to one list 
    non_anon_indexes = [item for sublist in non_anon_indexes for item in sublist]
    print('non_anon_indexes:', non_anon_indexes)
    fcol = 'force_anon_txt'
    df[fcol] = df[col]
    if curr_k < k:
        # Collecting the relevant BoW vectors
        non_anonym_vecs = []
        idx_list = []
        non_anonym_docs = []
        idx2 = 0
        for idx in range(len(df[col])):
            if idx in non_anon_indexes:
                non_anonym_vecs.append(vecs.toarray()[idx])
                idx_list.append(idx)
                idx2 += 0
                non_anonym_docs.append(df[col][idx])
        
        print('idx_list', idx_list)
        # Finding nearest k neighbors
        #neighbor_list = get_nearest_neighbors(non_anonym_vecs, k=k)
        #print('neighbor_list:', neighbor_list[:10])
        neighbor_list = get_nearest_neighbors(non_anonym_docs, k=k)
        print('neighbor_list:', neighbor_list[:10])

        # Replacing with *
        for idx1, n in enumerate(neighbor_list):
            print('\n n:', n)
            # Removing documents without partners
            if len(n) < k:
                for d in n:
                    df.loc[d, fcol] = '*'
                    print('After:', df.loc[d, fcol])
            else:
                # From indexes to vecors
                neighbor_vecs = [non_anonym_vecs[i] for i in n]
                diff = get_diff(neighbor_vecs)
                words_to_delete = voc[diff > 0]
                print('words_to_delete', words_to_delete)
                for d in n:
                    print('Before:', df.loc[d, fcol])
                    idx2 = idx_list[d]
                    for word in words_to_delete:
                        df.loc[d, fcol] = df.loc[idx2, fcol].replace(word, '*')
                    print('After:', df.loc[d, fcol])
    return df

def force_anonym_by_iteration(docs, k):
    """
    Method 2
    """
    vecs, voc = get_bow(docs)
    mat = vecs.toarray()
    # Finding the most rare words
    mat_sum = mat.sum(axis=0)
    rare_idx = mat_sum.argsort()
    
    curr_k, _ = get_anonym_degree(docs=docs)
    
    i = 0
    while (curr_k < k) and (i < len(rare_idx)):
        # Replacing the most rare word
        rword = voc[rare_idx[i]]
        print('Replace', rword)
        docs = [re.sub(fr'\b{rword}\b', '*', d) for d in docs]
        #docs = [d.replace(rword, '*') for d in docs]
        #print('docs:', docs)
        curr_k, _ = get_anonym_degree(docs=docs)
        print('curr_k', curr_k)

        if i == len(rare_idx):
            print('Replaced all words. Stopping')
        i += 1
    return docs


if __name__ == 'main':
    print('Hi')

    diff, comm = get_diff_and_common('hi i am hadas and i love banana', 'hi he is john and he loves hummus')