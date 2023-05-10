# Imports
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import spacy
import re
import bz2
from collections import defaultdict
import operator
from . import models

# nltk.download('stopwords')

stopword_list = stopwords.words('english')


# Defining some global variables
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])  # disabling Named Entity Recognition for speed
glove_model = models.glove_model


def corpus_stop_words(corpus,num_stop):
    """this function gets as input the text corpus (corpus) - list of texts, and a number (num_stop),
    it finds the most common words in the corpus and returns them as a dataframe/csv"""
    # Step 1: Tokenize the texts - make sure the corpus was pre-processed!! (lower/etc)
    tokenized_corpus = [text.split() for text in corpus]

    # Step 2: Count unique words per document
    unique_words_per_document = [set(text) for text in tokenized_corpus]

    # Step 3: Count word frequencies
    word_frequency = defaultdict(int)
    for words in unique_words_per_document:
        for word in words:
            word_frequency[word] += 1

    # Step 4: Sort and select top words
    most_frequent_words = sorted(word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_stop]

    return most_frequent_words # list of most frequent words to set as stop-words




def add_word_list_to_stop_words(filename):
    """
    Create a list of words from file
    """
    global stopword_list
    # Opening file in read mode
    with open(filename, 'r') as file:
        
        # reading the file
        data = file.read()
        
        # replacing end splitting the text 
        # when newline ('\n') is seen.
        data_into_list = data.split("\n")
        stopword_list = list(set(stopword_list + data_into_list))


def reading_bz_file(train_file):
    """ Reading the input file and returns a dataframe """
    # Credit: https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial

    # Readling the file to list of comments
    train_file = bz2.BZ2File(train_file)
    train_file_lines = train_file.readlines()

    # Converting from raw binary strings to strings that can be parsed
    train_file_lines = [x.decode('utf-8') for x in train_file_lines]

    # Extracting the labels and sentences
    train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
    train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]  # And converting to lower case

    # Converting to df
    df = pd.DataFrame(list(zip(train_sentences, train_labels)), columns=['txt', 'sentiment'])

    # Adding number of words
    df['num_of_words'] = df['txt'].apply(lambda x: len(x.split(' ')))

    return df


def jaccard_index(sentence1, sentence2):
    """ Calc Jaccard index for each pair of sentences """
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    if union > 0:
        jaccard = intersection / union
    else:
        print('Warning: jaccard_index union = 0', sentence1, sentence2)
        jaccard = 0
    return jaccard


def get_voc(corpus):
    """ Gets corpus voccabulary """
    word_set = set([])

    for doc in corpus:
        doc = doc.split(' ')
        for word in doc:
            word_set.add(word)
    return word_set


def get_lemma(word, nlp=nlp):
    """ Returns the lemma of the given word """
    word = nlp(word)
    txt = [token.lemma_ for token in word]
    lemma = txt[0]
    return lemma


def create_word_dict(corpus):
    """ Creates a word dictionary. Keys = words. Values = three boolians - stop/lemma/replaced"""

    all_words = get_voc(corpus)

    all_word_dict = {}

    for word in all_words:
        # Cleaning non AB characters
        word = replace_non_ab_chars(word)
        word_dict = {
            'protected': None,
            'lemma': None,
            'replaced': None}
        if len(word) > 0:
            if word in stopword_list:
                word_dict['protected'] = True
            elif get_lemma(word) != word:
                word_dict['lemma'] = get_lemma(word)
            all_word_dict[word] = word_dict

    return all_word_dict


def print_doc(doc, all_word_dict):
    """ Prints document based on the word dictionary """
    # out_str = 'Legend: (protected) [replaced] {lemmatized}\n'
    out_str = ''
    words = doc.split()

    for w in words:
        print_w = w.lower()
        w = replace_non_ab_chars(w)
        if w in all_word_dict:
            if all_word_dict[w]['protected']:
                print_w = '(' + print_w + ')'
            elif all_word_dict[w]['lemma']:
                print_w = '{' + all_word_dict[w]['lemma'] + '}'
            if all_word_dict[w]['replaced']:
                print_w = '[' + all_word_dict[w]['replaced'] +']'
        out_str = f'{out_str}{print_w} '

    return out_str


def replace_non_ab_chars(word):
    word = re.sub("[^A-Za-z']+", '', str(word)).lower()
    return word



def cleaning(doc, break_doc=False):
    """Lemmatizing and removes stopwords"""
    # Defining the document
    if break_doc:
        doc = doc.replace(' ', '. ' )
    doc = nlp(doc)

    # Lemmatizes and removes stopwords

    # For some reason, sometimes the lemmatization is not consistent - 
    # identical words receive a different lemmatization (for example, acting is sometimes 
    # changed to act and sometimes remains acting). 
    # Apperantly, Spacy's lemmatization depends on the part of speech: 
    # https://stackoverflow.com/a/74176351
    # To overcome this, we added the break_doc parameter

    txt = []
    for token in doc:
        word = re.sub('\W+', '', str(token)).lower()
        if word and (word not in stopword_list):
            txt.append(token.lemma_)

    # txt = [token.lemma_ for token in doc if (str(token) not in stopword_list)]

    clean_doc = ' '.join(txt)
    return clean_doc


def lemmatize_doc(doc, break_doc=False):
    """Lemmatizes document"""
    # Breaking the document to allow consistent lemmatization (see function cleaning())
    if break_doc:
        doc = doc.replace(' ', '. ' )
    doc = nlp(doc)

    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = []
    for token in doc:
        if str(token) in stopword_list:  # Add stopword as it is
            txt.append(str(token))
        else:  # Lemmatize other words
            txt.append(token.lemma_)

    clean_doc = ' '.join(txt)
    return clean_doc


def clean_corpus(corpus):
    """ Cleans the corpus """
    brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in corpus)
    corpus_lemmas = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]
    return corpus_lemmas


def get_average_jaccard(corpus, k=1):
    """ Calculates the avergae Jaccard index by averaging each documents k nearest neighbors """

    cc = clean_corpus(corpus)

    all_neighbors = []
    for idx1, doc1 in enumerate(cc):
        doc1_neighbors = []
        for idx2, doc2 in enumerate(cc):
            # Avod repeated comparisons
            if idx2 > idx1:
                doc1_neighbors.append(jaccard_index(doc1, doc2))
        doc1_neighbors.sort(reverse=True)
        all_neighbors += doc1_neighbors[:k]
    avg = np.average(all_neighbors)
    return avg


def plot_jaccard_hist(df_short_sentences, column = 'txt'):
    """creat a hist of jaccard scores"""

    indices_list = list(df_short_sentences.index)
    indices_list_short_1 = indices_list[1:4000]

    # Create a list of sentence texts
    sentences = list(df_short_sentences[column].loc[indices_list_short_1])

    # Compute the Jaccard index for all pairs of sentences
    jaccard_index_dict = {(indices_list_short_1[i], indices_list_short_1[j]): jaccard_index(sentences[i], sentences[j])
                          for i in range(len(sentences)) for j in range(i + 1, len(sentences))}

    # Sort the dictionary by its values in descending order
    sorted_dict = dict(sorted(jaccard_index_dict.items(), key=lambda item: item[1], reverse=True))

    # extract the values from the dictionary
    dic_values = list(sorted_dict.values())

    # plot a histogram of the values
    plt.hist(dic_values, bins=50, range=(0, 0.3))

    # set labels and title
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.title('Histogram of Jaccard Indecs')

    # display the plot
    plt.show()


def get_general_word_from_cluster(word_list, we_model):
    """ Finds the most similar words usind word embedding"""
    glove_words = list(we_model.index_to_key)
    known_words = [w for w in word_list if w in glove_words]
    if len(known_words) > 0:
        we_word = we_model.most_similar(known_words, topn=1)[0][0]
    else:
        we_word = None
    return we_word


def add_general_word_to_word_dict(word_dict, word):
    """ Updating that the given words were replaced """
    word_dict[word] = {
        'protected': False,
        'lemma': False,
        'replaced': True}
    return word_dict


def replace_words_in_df(df_0, cluster_dict, distance_dict, threshold, word_dict_0):
    """ Replaces the words in the dataframe """

    # Working with a copy of the df:
    df_copy = df_0.copy()
    word_dict_copy = word_dict_0.copy()

    df_copy['anon_txt'] = df_copy['txt'].apply(lambda x: lemmatize_doc(x))

    # create a list of the "new" words and don't cluster them in the next round
    new_words = []
    jacc_indexes = []
    k = 1

    start_jacc_index = get_average_jaccard(df_copy['anon_txt'], k=k)
    print('Starting average Jaccard index:', start_jacc_index)
    print('Distance threshold:', threshold)

    for key, words in cluster_dict.items():
        if key >= 0:  # Ignoring the -1 label
            #if len(cluster_dict[key]) < 20:  # so it will make sense!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if distance_dict[key] <= threshold:
                # Getting the general word
                general_word = get_general_word_from_cluster(words, glove_model)

                new_words.append(general_word)  # the list of new words
                print('distance:', distance_dict[key], '\treplacing', words, 'in', general_word)
                for word in words:
                    if word not in word_dict_copy:  # Lemmatized word
                        word_dict_copy[word] = {'protected': False, 'lemma': word, 'replaced':False}
                    if not word_dict_copy[word]['protected']:
                        # Updaing the word dictionary that the words were replaced
                        word_dict_copy[word]['replaced'] = general_word  # Problem: greate in line 2 miss-identified as replaced

                        # Replacing whole words
                        df_copy['anon_txt'] = df_copy['anon_txt'].replace(fr'\b{word}\b', general_word, regex=True)
            else:
                print('distance:', distance_dict[key],'the next cluster is too wide and wont be replaced:', cluster_dict[key])

        # Checking current average Jaccard distance
        curr_jacc_index = get_average_jaccard(df_copy['anon_txt'], k=k)
        jacc_indexes.append(curr_jacc_index)

    print('Final average Jaccard index:', get_average_jaccard(df_copy['anon_txt'], k=k))
    df_copy['anon_txt_history'] = df_copy['txt'].apply(lambda x: print_doc(x, word_dict_copy))
    df_copy['num_replaced'] = df_copy['anon_txt_history'].apply(lambda x: len(re.findall(r'\[\w+\]', x)))
    df_copy['num_lemmatized'] = df_copy['anon_txt_history'].apply(lambda x: len(re.findall(r'\{\w+\}', x)))
    df_copy['num_protected'] = df_copy['anon_txt_history'].apply(lambda x: len(re.findall(r'\(\w+\)', x)))
    df_copy['num_no_change'] = df_copy['num_of_words'] - df_copy['num_replaced'] - df_copy['num_lemmatized'] - df_copy['num_protected']

    # Plotting
    plt.plot(jacc_indexes)
    plt.xlabel('# of replaced clusters')
    plt.ylabel('Average Jaccard index')

    return df_copy, word_dict_copy


def get_stat(word_dict):
    """
    Returns the number of stop words, lemmatized and replaced words
    """
    prot_count = 0
    lemm_count = 0
    rep_count = 0

    all = 0

    for word in word_dict.values():
        if word['protected']:
            prot_count += 1
        elif word['replaced']:
            rep_count += 1
        elif word['lemma']:
            lemm_count += 1
        all += 1
    return prot_count, lemm_count, rep_count, all


if __name__ == 'main':
    print('yehhh')

