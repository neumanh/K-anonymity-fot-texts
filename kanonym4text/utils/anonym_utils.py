import numpy as np
import pandas as pd
import scipy
import re
import logging

from sklearn.feature_extraction.text import CountVectorizer
from annoy import AnnoyIndex
from k_means_constrained import KMeansConstrained
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from . import nlp_utils


def reduce_dims(vecs, dims=100):
    """
    Reduces dimensions using PCA.
    Returns: the low-dimensional vectors
    """
    # Reduce dimensions only if the dimensions are big
    if dims < min(vecs.shape):
        pca = PCA(n_components=dims)
        vecs = pca.fit_transform(vecs)
    return vecs


def lemmatize_and_remove_stops(corpus, stop_list):
    """
    Alternative version to nlp_utils.clean_corpus
    """
    c_corpus = []
    for doc in corpus:
        c_doc = nlp_utils.cleaning(doc, stop_list)
        c_corpus.append(c_doc)
        
    return c_corpus


def remove_stops(corpus, stop_list):
    """
    remove stop words only
    """
    c_corpus = []
    for doc in corpus:
        if doc:
            doc_tokenize = nlp_utils.nlp(doc)
            txt = []
            for token in doc_tokenize:
                word = re.sub('\W+', '', str(token)).lower()
                if word and (word not in stop_list):
                    txt.append(str(token))  # include all the marks that are not words
            clean_doc = ' '.join(txt)
        else:
            clean_doc = doc
        c_corpus.append(clean_doc)
    return c_corpus


def get_bow(corpus, stop_list, lemmatize=True):
    """ Vectorize the corpus using CountVectorizer """
    
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    if stop_list is not None:
        if lemmatize:
            cc = lemmatize_and_remove_stops(corpus, stop_list)
        else:
            cc = remove_stops(corpus, stop_list)
    else:
        logging.warning('Creating BoW without stopwords removal')
        cc = corpus
    try:
        count_data = vectorizer.fit_transform(cc)
        voc = vectorizer.get_feature_names_out()

    except Exception as e:
        logging.info(f'Could not create a bow: {e}')
        count_data, voc = None, None
    return count_data, voc


def get_tfidf(corpus):  # stop_list, clean = True, lemmatize = True):
    """ Vectorize the corpus using CountVectorizer """

    try:
        tfidf_model = TfidfVectorizer()

        # Fit Model
        tfidf_vectors = tfidf_model.fit_transform(corpus)
        # voc = tfidf_model.get_feature_names_out()
        voc = None  # Not needed

    except Exception as e:
        logging.warning('Could not create a Tf-Idf:', e)
        tfidf_model, voc = None, None
    
    return tfidf_vectors, voc


def get_anonym_degree(docs=None, vecs=None, min_k=None, stop_list=None):
    """ If K not given, returns the minimal current k and the corresponding documents.
        If k is given, return the documents with k or less neighbohrs  """
    
    if docs is not None:
        # Lemmatizing the documents
        count_data, _ = get_bow(docs, stop_list=stop_list, lemmatize=False)
    elif vecs is not None:
        count_data = vecs
    else:  # No input docs or vecs
        print('You must supply documents or vectors')
        return
    if count_data is not None:
        # Converting to array if is sparse
        if scipy.sparse.issparse(count_data):
            count_data = count_data.toarray()
        # Converting any number larger than 1 into 1
        count_data[count_data > 1] = 1
        # Counting unique values
        uniq_arr, uniq_cnt = np.unique(count_data, axis=0, return_counts=True)

        if not min_k:
            min_k = min(uniq_cnt)
            # All the unique vectors
            un_anon = uniq_arr[uniq_cnt <= min_k]
        else:
            # All the unique vectors
            un_anon = uniq_arr[uniq_cnt < min_k]
            min_k = min(uniq_cnt)  # For the return value

        # Getting the unique vector indices
        indices_list = []
        for row in un_anon:
            # Get the similar rows
            similar_vals = np.where((count_data == row).all(axis=1))
            indices_list.append(similar_vals[0].tolist())
    else:  # count_data is None
        min_k, indices_list = None, []

    return min_k, indices_list


def get_diff(vecs):
    """
    Finds the differences between arrays of 0,1.
    The input is either a list of lists or a matrix
    """
    # Creating a matrix if needed
    if isinstance(vecs, list):
        mat = np.vstack(vecs)
    else: 
        mat = vecs
    mat[mat > 0] = 1
    xsum = np.sum(mat, axis=0)

    union_x = (xsum > 0).astype('int')
    inter_x = (xsum == mat.shape[0]).astype('int')

    diff = union_x - inter_x
    return diff


def build_annoy_search_tree(n_list):
    """
    Builds annoy search tree
    n_list: A matrix. rows = documents. columns = vocabulary
    """
    # Build Annoy index with 10 trees. angular = cosine similarity
    annoy_index = AnnoyIndex(n_list.shape[1], metric='angular')
    for i, x in enumerate(n_list):
        annoy_index.add_item(i, x)
    annoy_index.build(10)

    return annoy_index


def get_k_unused_items(item, annoy_tree, used_items, k):
    """
    Returns k unused items.
    """
    # Find the k nearest neighbors to the first sentence
    k_unused_items = []
    new_k = k

    i = 0  # TEMP
    while len(k_unused_items) < k:
        nearest_neighbors = annoy_tree.get_nns_by_vector(item, new_k)
        for nn in nearest_neighbors:
            if (nn not in used_items) and (len(k_unused_items) < k) and (nn not in k_unused_items):
                k_unused_items.append(nn)
        new_k += 1
        i += 1  # TEMP
    return k_unused_items


def find_k_neighbors_using_annoy(docs, k, dim_reduct=0, stop_list = None):
    """
    1. Create BoW representation
    2. For each document, finds k nearest neighbors
    Returns a list of indexes.
    """
    if isinstance(docs, np.ndarray):  # Documents already embedded
        vecs = docs
    
    else:  # A list of texts
        # Cleaning the documents
        cdocs = [nlp_utils.lemmatize_doc(doc, stop_list=stop_list) for doc in docs]
        docs = cdocs
        vecs, _ = get_tfidf(docs)
        vecs = vecs.toarray()  # From sparse matrix to array 
        
        # Reduce dimensions
        if dim_reduct:
            vecs = reduce_dims(vecs=vecs, dims=dim_reduct)

    neighbor_list = []
    annoy_tree = build_annoy_search_tree(vecs)
    logging.debug('Finding k neighbors using Annoy...')

    used_indexes = set([])
    num_docs = vecs.shape[0]
    for i in range(num_docs):
        # To prevent redundant
        if i not in used_indexes:
            similar_doc_ind = get_k_unused_items(vecs[i], annoy_tree, used_indexes, k)
            neighbor_list.append(similar_doc_ind)
            for sd in similar_doc_ind:
                if sd in used_indexes:
                    logging.warning(f'The index {sd} was already used.')
                # Adding the index to the used items
                used_indexes.add(sd)  
            # No more k unused indexes
            if len(used_indexes) > (num_docs - k):
                break

    return neighbor_list


def reorder_documents(doc_list, neighbor_list, num):
    """
    Re-orders documents after anonymization.
    Returns a list of documents in the original order
    """
    # Creating a list with n Nones
    anonym_docs = [None] * num

    for indexes, docs in zip(neighbor_list, doc_list):
        for i, d in zip(indexes, docs):
            anonym_docs[i] = d
    return anonym_docs


def create_neighbors_df_for_comparison(doc_list, neighbor_list):
    """
    Saves the neighbors in a dataframe
    """
    neighbor_list_docs = [None] * len(neighbor_list)
    for i, nl in enumerate(neighbor_list):
        neighbor_list_docs[i] = list(neighbor_list[i])
        for j, doc_idx in enumerate(nl):
            neighbor_list_docs[i][j] = doc_list[doc_idx]
    return pd.DataFrame(neighbor_list_docs)


def force_anonym(docs, neighbor_list, stop_list):
    """ 
    For each group of neighbors, replace different words in *.
    Returns a list of anonymize documents.
    """
    annon_docs = docs.copy()
 
    cdocs = [nlp_utils.lemmatize_doc(doc, stop_list=stop_list) for doc in docs]
    docs = cdocs

    used_indexes = []
    for neighbors in neighbor_list:
        # print('neighbors:', neighbors)
        curr_docs = []
        for n in neighbors:
            # Adding the document to the similar doc list
            curr_docs.append(docs[n])
        anonym_docs = delete_uncommon_words(curr_docs, stop_list)
        i = 0
        for n in neighbors:
            used_indexes += neighbors
            annon_docs[n] = anonym_docs[i]
            i += 1

    # Removing all the documents without partners
    unused_indexes = list(set(range(len(docs))) - set(used_indexes))
    for i in unused_indexes:
        annon_docs[i] = '*'

    return annon_docs


def add_neighbor_list_to_df(df, neighbor_list):
    """
    Adds the neighbors for each document
    """
    df['neighbors'] = None  # corrected typo in column name from "neigbors" to "neighbors"
    for k_neighbors in neighbor_list:
        for n in k_neighbors:
            df.loc[n, 'neighbors'] = str(k_neighbors)
    return df


def delete_uncommon_words(docs, stop_list):
    """
    Deletes the uncommon words from given documents
    """
    # Lemmatizing
    ldocs = [nlp_utils.lemmatize_doc(d, stop_list=stop_list) for d in docs]

    vecs, voc = get_bow(ldocs, stop_list=stop_list)

    if vecs is not None:
        vecs = vecs.toarray()
        diff = get_diff(vecs)

        words_to_delete = voc[diff > 0]
        
        # logging.debug(f'docs: {docs} \nuncommon words: {words_to_delete}')

        temp_docs = []
        for d in ldocs:
            new_d = d
            for word in words_to_delete:
                # new_d = new_d.replace(word, '*')
                new_d = re.sub(rf'\b{word}\b', '*', new_d, flags=re.IGNORECASE)
            temp_docs.append(new_d)
    else:
        # All stop words. Return lemmatized document
        temp_docs = ldocs
    
    # logging.debug(f'After forcing: {temp_docs}')
    return temp_docs


def delete_word(word_dict, word):
    """
    When forcing anonymization, finds the word in the dictionary, and updates the replaced tag.
    """
    if word in word_dict:
        word_dict[word]['replaced'] = '*'
    else:
        for w in word_dict.keys():
            if (word_dict[w]['lemma'] == word) or (word_dict[w]['replaced'] == word):
                word_dict[w]['replaced'] = '*'
                break  # Exit for


# TEMP - also in LLM-utils
def ckmeans_clustering(docs, k, n_jobs = -1, dim_reduct = 0, stop_list = None):
    """
    Runs k-means with constrains.
    More on constrain-k-means:
    https://towardsdatascience.com/advanced-k-means-controlling-groups-sizes-
    and-selecting-features-a998df7e6745
    Returns a list of neighbor list
    """
    logging.debug(f'{len(docs)} documents')
    
    if isinstance(docs, np.ndarray):  # Documents already embedded
        vecs = docs
    else:  # A list of texts
        # Cleaning the documents
        cdocs = [nlp_utils.lemmatize_doc(doc, stop_list=stop_list) for doc in docs]
        docs = cdocs
        vecs, _ = get_tfidf(docs)

        vecs = vecs.toarray()  # From sparse matrix to array (KMeansConstrained does not work on sparse matrix)

    # Reduce dimensions
    if dim_reduct:
        vecs = reduce_dims(vecs=vecs, dims=dim_reduct)

    # logging.info('Got BoW')
    num_clusters = vecs.shape[0] // k
    min_size = k
    max_size = k

    # For example, if k=3 and there are 100 sequences,
    # allow one cluster with k+1
    mod_data =  vecs.shape[0] % k
    if mod_data != 0:
        max_size += mod_data

    # Clustering parameters
    n_init, max_iter = 6, 100
    
    clf = KMeansConstrained(
     n_clusters=num_clusters,
     size_min=min_size,
     size_max=max_size,
     random_state=0,
     n_jobs=n_jobs,
     n_init=n_init,
     max_iter=max_iter
    )
    # Logging
    logging.debug(f'Clustering... Data dimensions: {vecs.shape}, Parameters: n_init {n_init} '
                  f'  max_iter {max_iter}   n_jobs {n_jobs}')

    try:
        clf.fit_predict(vecs)
        logging.debug('Finish clustering. Getting original indexes.')
        pair_list = []
        for i in range(1, num_clusters):
            curr_pair = np.where(clf.labels_ == (i))[0].tolist()
            if curr_pair not in pair_list:
                pair_list.append(tuple(curr_pair))
    except Exception as e:
        logging.error(f'Could not find neighbors using k_means_constrained: {e}')
        logging.info('Finding K neighbors using Annoy')
        # Sending the already cleaned vectors
        pair_list = find_k_neighbors_using_annoy(docs=None, k=k, vecs=vecs, dim_reduct=dim_reduct)
        
    return pair_list
