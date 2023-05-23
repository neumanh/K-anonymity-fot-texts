import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import pdist, squareform
import re
from annoy import AnnoyIndex
import scipy

from . import nlp_utils

# CountVectorizer is defined only once
vectorizer = CountVectorizer(ngram_range=(1,1)) # to use bigrams ngram_range=(2,2)
#                           stop_words='english')

def lemmatize_and_remove_stops(corpus, break_doc=False):
    """
    Alternative version to nlp_utils.clean_corpus
    """
    ccorpus = []
    for doc in corpus:
        cdoc = nlp_utils.cleaning(doc, break_doc)
        ccorpus.append(cdoc)
        
    return ccorpus

def remove_stops(corpus):
    """
    remove stop words only
    """
    ccorpus = []
    for doc in corpus:
        if doc:
            doc_tokenize = nlp_utils.nlp(doc)
            #print("doc")
            #print(doc)

            #print("doc_tokenize")
            #print(doc_tokenize)

            txt = []
            for token in doc_tokenize:
                word = re.sub('\W+', '', str(token)).lower()
                # print("token",token,"word",word)

                if word and (word not in nlp_utils.stopword_list):
                    txt.append(str(token)) # onclude all the marks that are not words 
        
            clean_doc = ' '.join(txt)

        else:
            clean_doc = doc
        ccorpus.append(clean_doc)

    return ccorpus



        
    return ccorpus

def get_bow(corpus, break_doc = False, lemmatize = True):
    """ Vectorizes the corpus using CountVectorizer """
    if lemmatize:
        cc = lemmatize_and_remove_stops(corpus, break_doc=break_doc)
    else:
        cc = remove_stops(corpus)
    # print(corpus)
    # print(cc)

    try:
        # Vectorizing
        count_data = vectorizer.fit_transform(cc)
        voc = vectorizer.get_feature_names_out()

    except Exception as e:
        print('Could not create a bow:', e)
        count_data, voc = None, None
    return count_data, voc


def get_anonym_degree(docs = None, vecs = None, min_k = None):
    """ If K not given, returns the minimal current k and the corresponding documents.
        If k is given, return the documents with k or less neighbohrs  """
    
    if docs is not None:
        # Lemmatizing the documents
        count_data, voc = get_bow(docs, break_doc = False,lemmatize = False)
    elif vecs is not None:
        count_data = vecs
    else: # No input docs or vecs
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
            min_k = min(uniq_cnt) # For the return value

        # Getting the unique vectore indeces
        indeces_list = []
        for row in un_anon:
            # Get the similar rows
            similar_vals = np.where((count_data == (row)).all(axis=1))
            indeces_list.append(similar_vals[0].tolist())
    else:  # count_data is None
        min_k, indeces_list = None, None

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
    doc = nlp_utils.cleaning(doc)
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
    The input is either a list of lists or a matrix
    """
    # Creating a matrix if needed
    if isinstance(vecs, list):
        mat = np.vstack(vecs)
    else: 
        mat = vecs
    mat[mat > 0] = 1
    #nparray = sparse_mat.toarray()
    xsum = np.sum(mat, axis=0)

    #print('xsum\n', xsum)

    union_x = (xsum > 0).astype('int')
    inter_x = (xsum == mat.shape[0]).astype('int')

    diff = union_x - inter_x
    return diff


def jaccard_index(u, v):
    """
    For vectors of 0 and 1
    Creadit: JDWarner https://gist.github.com/JDWarner/6730886
    """
    u[u > 1] = 1
    v[v > 1] = 1
    if np.double(np.bitwise_or(u, v).sum()) != 0:
        j = np.double(np.bitwise_and(u, v).sum()) / np.double(np.bitwise_or(u, v).sum())
    else:
        j = 0

    return j


def build_annoy_search_tree(n_list):
    """
    Builds annoy search tree
    n_list: A matrix. rows = documents. columns = vocabulary
    """
    # Build an Annoy index with 10 trees. angular = cosine similarity
    annoy_index = AnnoyIndex(n_list.shape[1], metric='angular')
    for i, x in enumerate(n_list):
        annoy_index.add_item(i, x)
    annoy_index.build(10)

    return annoy_index
    

def get_nearest_neighbors_annoy(item, n_list, k):
    """ 
    Find the K nearest neighbors using Annoy package and cosine similarity.
    item is the single vector to search
    n_list is a numpy matrix
    k is the number of neighbors
    """

    # Build an Annoy index with 10 trees. angular = cosine similarity
    annoy_index = AnnoyIndex(n_list.shape[1], metric='angular')
    for i, x in enumerate(n_list):
        annoy_index.add_item(i, x)
    annoy_index.build(10)

    # Find the k nearest neighbors to the first sentence
    nearest_neighbors = annoy_index.get_nns_by_vector(item, k)

    return nearest_neighbors


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


def force_anonym_using_annoy(docs, k):
    """ 
    Steps:
    1. Create BoW representation
    2. For each document, finds k nearest neighbors
    3. For each group of neigbors, replace different words in *
    """
    annon_docs = docs.copy()
    used_indexes = set([])

    cdocs = [nlp_utils.lemmatize_doc(doc) for doc in docs]
    docs = cdocs

    vecs, _ = get_bow(docs, lemmatize = True)
    vecs = vecs.toarray()  # From sparse matrix to array
    curr_k, non_anon_indexes = get_anonym_degree(vecs=vecs)
    print('Start: get_anonym_degree:', curr_k)
    
    temp_docs_emb = vecs.copy()

    neighbor_list = []
    if k>= curr_k: # if i already curr_k than don't run the following:

        annoy_tree = build_annoy_search_tree(vecs)

        for i, _ in enumerate(docs):
            #print('i:', i, '\t', used_indexes)
            # To prevent redandent
            if i not in used_indexes:
                similar_doc_ind = get_k_unused_items(vecs[i], annoy_tree, used_indexes, k)
                #used_indexes.add(i)  # Adding to the used items
                # similar_doc_ind = get_nearest_neighbors_annoy(temp_docs_emb[i], temp_docs_emb, k)
                neighbor_list.append(similar_doc_ind)
                print('similar_doc_ind', similar_doc_ind)
                curr_docs = []
                for sd in similar_doc_ind:
                    if sd in used_indexes:
                        print('Error!:', sd, 'was already used.')
                    # Adding the document to the similar doc list
                    curr_docs.append(docs[sd])
                    # Adding the index to the used items
                    used_indexes.add(sd)  
                    # Prevent repeating comparison by changing the vector
                    # temp_docs_emb[sd] = (-1000) * np.random.randint(10, size=len(temp_docs_emb[sd]))
                    #temp_docs_emb[sd] = [1000] * len(temp_docs_emb[sd])
                anonym_docs = delete_uncommon_words(curr_docs)
                i = 0
                for sd in similar_doc_ind:
                    annon_docs[sd] = anonym_docs[i]
                    i += 1
            if  len(used_indexes) > (len(docs) - k):
                print('Breaking after moving over', len(used_indexes), 'of all', len(docs), 'indexes.')
                #print('Breaking! \tlen(used_indexes)', len(used_indexes), '\tlen(docs)', len(docs), '\tlen(docs)-k', (len(docs) - k))
                # Deleting the remaining docs
                unused_indexes = list(set(range(len(docs))) - set(used_indexes))
                print('unused_indexes:', unused_indexes)
                for i in unused_indexes:
                    annon_docs[i] = '*'
                break
        curr_k, _ = get_anonym_degree(docs=annon_docs)
        print('End: get_anonym_degree:', curr_k) 
    else:
        annon_docs = None
        neighbor_list = None
        print(f"we already have k-anonymity for k={k}")
    return annon_docs, neighbor_list


def add_neighbor_list_to_df(df, neighbor_list):
    """
    Adds the neigbors for each document
    """
    df['neigbors'] = None
    for k_neighbors in neighbor_list:
        for n in k_neighbors:
            df.loc[n, 'neigbors'] = str(k_neighbors)
    return df


def delete_uncommon_words(docs):
    """
    Deletes the uncommon words from given documents
    """
    # Lemmatizing
    ldocs = [nlp_utils.lemmatize_doc(d) for d in docs]

    vecs, voc = get_bow(ldocs)

    vecs = vecs.toarray()
    diff = get_diff(vecs)

    words_to_delete = voc[diff > 0]

    temp_docs = []
    for d in ldocs:
        new_d = d
        for word in words_to_delete:
            #new_d = new_d.replace(word, '*')
            new_d = re.sub(rf'\b{word}\b', '*', new_d)
        temp_docs.append(new_d)

    return temp_docs


def get_nearest_neighbors_using_jaccard(n_list, k):
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


if __name__ == 'main':
    import sys
    sys.path.append("utils")
    from utils import nlp_utils

    print('Hiiiiiiiiiiiiiii')
