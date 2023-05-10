# Imports
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
import matplotlib.pyplot as plt
#from nltk.corpus import stopwords
from itertools import combinations
import umap
# import hdbscan
import sklearn.cluster as cluster

from . import models
from . import nlp_utils

# upload model:
glove_model = models.glove_model

def define_max_threshold(glove_model=glove_model):
    """Defines distance between pairs of words"""
    
    # Collecting distances of good pairs
    sim_list_best = get_pairs_dist(get_good_pairs(), glove_model)
    print('sim_list_best', np.mean(sim_list_best))

    # Collecting distances of bad pairs
    sim_list_worst = get_pairs_dist(get_bad_pairs(), glove_model)
    print('sim_list_worst', np.mean(sim_list_worst))
    
    best_dist = 1 - np.mean(sim_list_best)
    worst_dist = 1 - np.mean(sim_list_worst)

    # The  threshold for cluster max_dist should be in the middle of the two values above = 0.1765
    threshold = (best_dist+worst_dist)/2
    return threshold


def get_good_pairs():
    """Returns pairs of similar words"""
    best_pairs_ls = [
        ['good','great'],
        ['dog','cat'],
        ['green','yellow'],
        ['dad','mom'],
        ['purchase','buy'],
        ['gift','present'],
        ['fast','quick'],
        ['big','huge'],
        ['item','product'],
        ['text','script'],
        ['john','julie'], # Adding names
        ['shirley','matthew'],
        ['joseph','smith'],
        ['horton','sylvia'],
        ['bonnie','henry'],
        ['paul','jr'],
        ['duncan','kelly']]
    return best_pairs_ls


def get_bad_pairs():
    """Returns pairs of unsimilar words"""
    # worst_pairs_ls = [
    #     ['good','bad'],
    #     ['thin','fat'],
    #     ['black','white'],
    #     ['tall','small'],
    #     ['boy','buy'],
    #     ['novel','old'],
    #     ['high','low'],
    #     ['hot','cold'],
    #     ['strong','week'],
    #     ['poor','rich']]
    worst_pairs_ls = [
        ['good','trip'],
        ['think','fat'],
        ['sister','white'],
        ['grammar','small'],
        ['boy','buy'],
        ['playstation','old'],
        ['cd','low'],
        ['battery','cold'],
        ['wonderful','check'],
        ['book','rich'],
        ['brother','unless'],
        ['john','ear'],
        ['sixty','stay'],
        ['barbara','wave'],
        ['buy','niece']]
    return worst_pairs_ls


def get_pairs_dist(pair_list, glove_model):
    """Embed each word in the pairs and returns the distance between thems"""
    sim_list = []
    for pair in pair_list:
        # calc cosine dist between w1 and w2
        emb1 = glove_model[pair[0]]
        emb2 = glove_model[pair[1]]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_list.append(similarity)
        
    return sim_list


def get_word_list_for_clustering(word_dict):
    """Lemmatizing and remove stop words"""
    word_list = []
    for key, val in word_dict.items():
        if (not val['protected']):  # Not protected 
            if val['lemma']:
                word_list.append(val['lemma'])
            else:
                word_list.append(key)
            
    return list(set(word_list))  # Remove duplicates 


def embed_corpus(word_list):
    """ Embeds the corpus using glove """
    word_index = get_word_index_for_clustering(word_list)

    # Iterate over your dictionary of words and embed them using GloVe
    embedded_dict = {}
    for word, idx in word_index.items():
        if word not in nlp_utils.stopword_list: # stopwords.words('english'):
            try:
                embedded_dict[word] = glove_model[word]
            except KeyError:
                # If the word is not in the GloVe vocabulary, assign a default embedding or skip it
                pass
    return embedded_dict


def find_eps_val(embeddings, cosine = False):
    """ Finds the EPS value for clustering """

    # Compute the k-distances for each point
    k = 10
    if cosine:
        neigh = NearestNeighbors(n_neighbors=k, metric = 'cosine')
    else:
        neigh = NearestNeighbors(n_neighbors=k)  # Using Euclidian distance
    neigh.fit(embeddings)
    distances, indices = neigh.kneighbors(embeddings)

    # Sort the distances and flatten them into a 1D array
    sorted_distances = np.sort(distances[:, k - 1], axis=None)

    # Plot the k-distance graph
    plt.plot(sorted_distances)
    plt.title('Optimizing epsilon value')

    # Find the elbow point
    kneedle = KneeLocator(range(len(sorted_distances)), sorted_distances, S=1.0, curve='concave', direction='increasing')
    eps = sorted_distances[kneedle.elbow]

    return eps


def run_clustering(embedded_dict, cosine = False):
    """ Runs clustering """
    # Convert to numpy array
    embeddings = np.array(list(embedded_dict.values()))

    eps = find_eps_val(embeddings, cosine=cosine)
    # Chose 3 a min words per cluster (maybe reduce to 2?) Maybe according to k
    if cosine:
        dbscan = DBSCAN(eps=eps, min_samples=2, metric = 'cosine') 
    else:
        dbscan = DBSCAN(eps=eps, min_samples=2)  # Using Euclidian distance

    dbscan.fit(embeddings)
    print('eps', eps)

    labels = dbscan.labels_
    clusters = {}
    for i, key in enumerate(embedded_dict.keys()):
        cluster = labels[i]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(key)
    
    # Getting the maximal distance in each cluster
    distance_dict = get_dist_dict(embedded_dict, clusters, labels)

    return clusters, distance_dict, labels

def find_max_dist(embeddings: dict):
    """Finds the pair of most distant words in the embedded dict and return the words and the similarity score"""
    words = list(embeddings.keys())
    pairs = combinations(words, 2)
    max_dist = -1
    closest_pair = None
    for pair in pairs:
        emb1 = embeddings[pair[0]]
        emb2 = embeddings[pair[1]]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        dist = 1 - similarity
        if dist >  max_dist:
            max_dist = dist
            closest_pair = pair
    return closest_pair, max_dist


def get_dist_dict_0(embedded_dict, clusters, labels):
    """Calculates the max distance fore each cluster and return a dicionary of cluster num: max. distance"""
    # init dict of dist:
    dist_dict = {}

    # for each cluster return the pair of words and max dist of the cluster:
    for ind in set(labels):
        filtered_dict = {k: v for k, v in embedded_dict.items() if k in clusters[ind]} # dict of embeddings of the words in the cluster

        #filtered_dict = {k: v for k, v in embedded_dict.items() if v in clusters[ind]}  
        _ ,max_dist = find_max_dist(filtered_dict) # find the two most dist words in the cluster
        dist_dict[ind] = max_dist
    return dist_dict


def get_dist_dict(embedded_dict, clusters, labels):
    """Calculates the max distance fore each cluster and return a dicionary of cluster num: max. distance"""
    # init dict of dist:
    dist_dict = {}

    # for each cluster return the pair of words and max dist of the cluster:
    for ind in set(labels):
        filtered_dict = {k: v for k, v in embedded_dict.items() if k in clusters[ind]} # dict of embeddings of the words in the cluster
        curr_vecs = list(filtered_dict.values())

        # Finding the centorid
        centroid = np.mean(curr_vecs, axis=0)
        curr_dis_list = []
        for v in curr_vecs:
            # Adding the absolut distance from centorid
            curr_dis_list.append(np.abs(v-centroid)) 
        dist = np.mean(curr_dis_list)
        dist_dict[ind] = dist
    return dist_dict


def plot_tsne(embedded_dict, labels):
    # Extract the embeddings from the embedded_dict and store them in a numpy array
    embeddings = np.array(list(embedded_dict.values()))

    # Perform t-SNE on the embeddings to reduce their dimensionality to 2
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot the 2D embeddings with different colors for each cluster
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels)
    plt.show()

def plot_tsne_2(embedded_dict, labels): # not inclusing cluster -1
    # Extract the embeddings from the embedded_dict and store them in a numpy array
    embeddings = np.array(list(embedded_dict.values()))

    # Perform t-SNE on the embeddings to reduce their dimensionality to 2
    tsne = TSNE(n_components=2, perplexity=300, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot the 2D embeddings with different colors for each cluster
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels)
    plt.show()

from sklearn.decomposition import PCA


def plot_pca(embedded_dict,labels):
    embeddings = np.array(list(embedded_dict.values()))

    # X is your embedding matrix, with shape (num_samples, embedding_dim)
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    X_pca = pca.transform(embeddings)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

def get_word_index_for_clustering(all_words):
    """ Uses tokenizer to get word indexes """
    word_index = {}
    i = 0
    for word in all_words:
        if word and (word not in nlp_utils.stopword_list): #  stopwords.words('english')):
            word_index[word] = i
            i += 1

    return word_index


def run_clustering_hdbscan(embedded_dict):
    """
    Cluster embedded words using UMAP and H-DBSCAN
    """
    embeddings = np.array(list(embedded_dict.values())) # the vectors as np.array
    # Running UMAP
    clusterable_embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,).fit_transform(embeddings)

    # Running HDBSCAN
    labels = hdbscan.HDBSCAN(
    min_samples=10,
    min_cluster_size=3).fit_predict(clusterable_embedding)

    # Collecting clustered words
    hd_clusters = {}
    for i, key in enumerate(embedded_dict.keys()):
        cluster = labels[i]
        if cluster not in hd_clusters:
            hd_clusters[cluster] = []
        hd_clusters[cluster].append(key)
    
    # Getting the maximal distance in each cluster
    distance_dict = get_dist_dict(embedded_dict, hd_clusters, labels)

    return hd_clusters, distance_dict, labels


def plot_cluster_size_distribution(clusters):
    """
    Plots cluster size distribution.
    Input: dictionary of key: [item1, item2, ...]
    """
    size_list = []
    for l in clusters.values():
        size_list.append(len(l))
    plt.hist(size_list, bins=80)
    plt.gca().set(title='Cluster size Distribution', ylabel='Frequency', xlabel='Cluster size')


def plot_cluster_size_distribution_2(clusters):  # PIE CHART
    """
    Plots cluster size distribution.
    Input: dictionary of key: [item1, item2, ...]
    """
    # NOT INCLUSING CLUSTER -1
    size_list = []
    copy_clusters = clusters
    del copy_clusters[-1]

    print(copy_clusters)
    for l in copy_clusters.values():
      #print(l)
      if l == -1:
        continue
      else:
        size_list.append(len(l))

    # cluster_sizes is a list or array containing the number of data points in each cluster
    plt.pie(size_list, labels=size_list)
    # plt.pie(size_list, labels=range(len(size_list)))

    plt.show()
    # cluster_sizes is a list or array containing the number of data points in each cluster
    plt.bar(range(len(size_list)), size_list)
    plt.xlabel('Cluster')
    plt.ylabel('Number of data points')
    plt.show()



if __name__ == '__main__':
    print('YEHH')
    define_max_threshold()

