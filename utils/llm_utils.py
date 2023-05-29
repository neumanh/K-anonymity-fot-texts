from k_means_constrained import KMeansConstrained
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np


def ckmeans_clustering(data, k):
    """
    Runs k-means with constrains.
    Credit: https://towardsdatascience.com/advanced-k-means-controlling-groups-sizes-and-selecting-features-a998df7e6745
    """
    num_clusters = len(data) // k
    min_size = k
    max_size = k

    # For example, if k=3 and there are 100 sequences,
    # allow one cluster with k+1
    mod_data = len(data) % k
    if mod_data != 0:
        max_size += mod_data
    
    clf = KMeansConstrained(
     n_clusters=num_clusters,
     size_min=min_size,
     size_max=max_size,
     random_state=0
    )
    clf.fit_predict(data)
    pair_list = []
    for i in range(1, num_clusters):
        curr_pair = np.where(clf.labels_ == (i))[0].tolist()
        if curr_pair not in pair_list:
            pair_list.append(tuple(curr_pair))
        
    return pair_list


def print_example(indexes, origina_docs, new_docs):
    print('Before:')
    for i in indexes:
        print(origina_docs[i])
    print('\nAfter:')
    for i in indexes:
        print(new_docs[i])


def sum_text(doc_list, tokenizer, gen_model):
    # define the input sentences
    #input_text = '. '.join(doc_list)
    input_text = ''
    i = 1
    for doc in doc_list:
       input_text = f'{input_text}{i}: {doc}. ' 
       i += 1

    # preprocess the input sentences
    input_ids = tokenizer.encode(f'summarize:' + input_text, return_tensors="pt")

    # generate the summary sentence
    output_ids = gen_model.generate(input_ids=input_ids, max_length=32, num_beams=4, early_stopping=True)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output


def run_anonymization_on_txt(docs, k):
    """ Finding K nearest neighbors and summarize them """
    
    # Defining models
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
    emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1')
    
    annon_docs = docs.copy()
    
    # Embedding
    docs_emb = emb_model.encode(docs)

    neighbor_list = ckmeans_clustering(docs_emb, k)
    for n in neighbor_list:
        # print('n:', n)
        curr_docs = []
        for doc_index in n:
            # Adding the document to the similar doc list
            curr_docs.append(docs[doc_index])
        sum_doc = sum_text(curr_docs, tokenizer, gen_model)
        # print('similar_doc_ind', n, '\tSummary:', sum_doc)
        for doc_index in n:
            annon_docs[doc_index] = sum_doc

    return annon_docs, neighbor_list
