# Imports
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import logging


def define_eps_cos(wemodel):
    """Defines distance between pairs of words"""
    
    # Collecting distances of good pairs - cosine similarity
    sim_list_best = get_pairs_sim_cos(get_good_pairs(), wemodel)
    # print('mean similarity between good pairs\t', np.median(sim_list_best))

    # Collecting distances of bad pairs - cosine similarity
    sim_list_worst = get_pairs_sim_cos(get_bad_pairs(), wemodel)
    # print('mean similarity between bad pairs\t', np.median(sim_list_worst))
    
    best_dist = 1 - np.median(sim_list_best)
    worst_dist = 1 - np.median(sim_list_worst)

    # The  threshold for cluster max_dist should be in the middle of the two values above = 0.1765
    threshold = (best_dist + worst_dist)/2
    
    return threshold


def define_eps_euc(wemodel):
    """Defines distance between pairs of words"""
    
    # Collecting distances of good pairs - Euclidean distance
    dist_list_best = get_pairs_dist_euc(get_good_pairs(), wemodel)
    # print('mean distance between good pairs\t', np.median(dist_list_best))

    # Collecting distances of bad pairs - Euclidean distance
    dist_list_worst = get_pairs_dist_euc(get_bad_pairs(), wemodel)
    # print('mean distance between bad pairs\t', np.median(dist_list_worst))
    
    best_dist = np.median(dist_list_best)
    worst_dist = np.median(dist_list_worst)

    # The  threshold for cluster max_dist should be in the middle of the two values above = 0.1765
    threshold = (best_dist + worst_dist)/2
    return threshold


def get_good_pairs():
    """Returns pairs of similar words"""
    best_pairs_ls = [
        ['good', 'great'],
        ['dog', 'cat'],
        ['green', 'yellow'],
        ['dad', 'mom'],
        ['purchase', 'buy'],
        ['gift', 'present'],
        ['fast', 'quick'],
        ['big', 'huge'],
        ['item', 'product'],
        ['text', 'script'],
        ['john', 'julie'],  # Adding names
        ['shirley', 'matthew'],
        ['joseph', 'smith'],
        ['michael', 'sylvia'],
        ['bonnie', 'henry'],
        ['paul', 'jr'],
        ['duncan', 'kelly']]
    return best_pairs_ls


def get_bad_pairs():
    """Returns pairs of dissimilar words"""
    worst_pairs_ls = [
        ['good', 'trip'],
        ['think', 'fat'],
        ['sister', 'white'],
        ['grammar', 'small'],
        ['boy', 'buy'],
        ['playstation', 'old'],
        ['cd', 'low'],
        ['battery', 'cold'],
        ['wonderful', 'check'],
        ['book', 'rich'],
        ['brother', 'unless'],
        ['john', 'ear'],
        ['sixty', 'stay'],
        ['barbara', 'wave'],
        ['buy', 'niece']]
    return worst_pairs_ls


def get_pairs_sim_cos(pair_list, wemodel):
    """Embed each word in the pairs and returns the distance between them"""
    sim_list = []
    for pair in pair_list:
        # calc cosine dist between w1 and w2
        emb1 = wemodel[pair[0]]
        emb2 = wemodel[pair[1]]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_list.append(similarity)
        
    return sim_list


def get_pairs_dist_euc(pair_list, wemodel):
    """Embed each word in the pairs and returns the distance between them"""
    dist_list = []
    for pair in pair_list:
        # calc cosine dist between w1 and w2
        emb1 = wemodel[pair[0]]
        emb2 = wemodel[pair[1]]
        dist = np.linalg.norm(emb1 - emb2)
        dist_list.append(dist)
        
    return dist_list


def get_word_list_for_clustering(word_dict):
    """Lemmatizing and remove stop words"""
    word_list = []
    for key, val in word_dict.items():
        if not val['protected']:  # Not protected
            if val['lemma']:
                word_list.append(val['lemma'])
            else:
                word_list.append(key)
            
    return list(set(word_list))  # Remove duplicates 


def embed_corpus(word_dict, stop_list, wemodel):
    """ Embeds the corpus using glove """

    word_list = get_word_list_for_clustering(word_dict)
    word_index = get_word_index_for_clustering(word_list, stop_list)

    # Iterate over your dictionary of words and embed them using GloVe
    embedded_dict = {}
    for word in word_index.keys():
        if word not in stop_list: # stopwords.words('english'):
            try:
                embedded_dict[word] = wemodel[word]
            except KeyError:
                # If the word is not in the GloVe vocabulary, assign a default embedding or skip it
                pass
    return embedded_dict


def run_clustering(word_dict, stop_list, wemodel, cosine=True, eps=None, n_jobs=-1):
    """ Runs clustering """
    # point to think - min_points in cluster to be defined according to k?
    # Get embedding
    embedded_dict = embed_corpus(word_dict, stop_list, wemodel)
    # Convert to numpy array
    embeddings = np.array(list(embedded_dict.values()))

    if not eps:
        # Based on embedding distance / 2, since DBSCAN uses the epsilon as radius and not diameter.
        if cosine:
            eps = define_eps_cos(wemodel=wemodel) / 2
        else:
            eps = define_eps_euc(wemodel=wemodel) / 2 
            # eps = find_eps_val(embeddings, cosine=cosine)  # Based on knee method

    logging.debug(f'Pre-defined epsilon for DBSCAN: {eps}.   Cosine metric: {cosine}.   n_jobs: {n_jobs}')
    
    # Chose 3 a min words per cluster (maybe reduce to 2?) Maybe according to k
    if cosine:
        dbscan = DBSCAN(eps=eps, min_samples=2, metric='cosine', n_jobs=n_jobs) 
    else:
        dbscan = DBSCAN(eps=eps, min_samples=2, n_jobs=n_jobs)  # Using Euclidian distance

    dbscan.fit(embeddings)

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


def get_dist_dict(embedded_dict, clusters, labels):
    """Calculates the max distance fore each cluster and return a dictionary of cluster num: max. distance"""
    # init dict of dist:
    dist_dict = {}

    # for each cluster return the pair of words and max dist of the cluster:
    for ind in set(labels):
        filtered_dict = {k: v for k, v in embedded_dict.items() if k in clusters[ind]}
        curr_vecs = list(filtered_dict.values())

        # Finding the centroid
        centroid = np.mean(curr_vecs, axis=0)
        curr_dis_list = []
        for v in curr_vecs:
            # Adding the absolut distance from centroid
            curr_dis_list.append(np.abs(v-centroid)) 
        dist = np.mean(curr_dis_list)
        dist_dict[ind] = dist
    return dist_dict


def get_word_index_for_clustering(all_words, stop_list):
    """ Uses tokenizer to get word indexes """
    word_index = {}
    i = 0
    for word in all_words:
        if word and (word not in stop_list):  # stopwords.words('english')):
            word_index[word] = i
            i += 1

    return word_index


def plot_cluster_size_distribution(clusters):  # PIE CHART
    """
    Plots cluster size distribution.
    Input: dictionary of key: [item1, item2, ...]
    """
    # NOT INCLUDING CLUSTER -1
    size_list = []
    copy_clusters = clusters
    if -1 in copy_clusters:
        del copy_clusters[-1]

    for l in copy_clusters.values():
      if l == -1:
        continue
      else:
        size_list.append(len(l))

    # cluster_sizes is a list or array containing the number of data points in each cluster
    plt.pie(size_list, labels=size_list)
    # plt.pie(size_list, labels=range(len(size_list)))

    plt.show()
    # cluster_sizes is a list or array containing the number of data points in each cluster
    # Set the maximum number of bins you want to display
    max_bins = 10

# Get the limited list of sizes based on the maximum number of bins
    limited_size_list = size_list[:max_bins]
    plt.bar(range(len(limited_size_list)), limited_size_list)
    # Customize the x-axis labels if needed
    plt.xticks(range(len(limited_size_list)), range(1, len(limited_size_list) + 1))
    plt.xlabel('Cluster No.')
    plt.ylabel('Words in cluster')
    plt.show()


if __name__ == '__main__':
    print('YEHH')

