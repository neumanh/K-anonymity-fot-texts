# Imports
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# import spacy
import bz2
from collections import defaultdict
import operator
import logging
import en_core_web_sm

# Defining some global variables
stopword_list = None

short_stopword_list = None
long_stopword_list = None

# nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])  # disabling Named Entity Recognition for speed
nlp = en_core_web_sm.load()

def get_list_from_file(file_name, num):
    """"
    Reads a file with words (each word on different line) and returns a list of these words.
    """
    try:
        with open(file_name, 'r') as file:
            
            # reading the file
            data = file.read()
            
            # replacing end splitting the text 
            # when newline ('\n') is seen.
            word_list = data.split("\n")

            # Take only part of the words
            if (num >= 0) and (num < len(word_list)):
                word_list = word_list[:num]
    
    except Exception as e:
        logging.error(f'Could not read the file {file_name}: {e}')
        word_list = None
    return word_list


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


def get_voc(corpus):
    """
    Creates a set of unique words in a given corpus
    """
    word_set = set([])

    for doc in corpus:
            doc_words = set(word_tokenize(doc))
            word_set = word_set.union(doc_words)
    return word_set


def get_lemma(word, nlp=nlp):
    """ Returns the lemma of the given word """
    word = nlp(word)
    txt = [token.lemma_ for token in word]
    lemma = txt[0]
    return lemma


def create_word_dict(corpus, stop_list):
    """ Creates a word dictionary. Keys = words. Values = three boolians - stop/lemma/replaced"""
    all_words = get_voc(corpus)

    logging.debug('Got vocabulary')

    all_word_dict = {}

    for word in all_words:
        # Cleaning non AB characters
        word = replace_non_ab_chars(word)

        # Creating the word dictionary
        word_dict = {
            'protected': None,
            'lemma': None,
            'replaced': None}
        if len(word) > 0:
            if word in stop_list:
                word_dict['protected'] = True
            else:
                lemma = get_lemma(word)
                if  lemma != word:
                    word_dict['lemma'] = lemma
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


def cleaning(doc, stop_list):
    """Lemmatizing and removes stopwords"""
    # Defining the document
    if doc:
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
            if word and (word not in stop_list):
                txt.append(token.lemma_)
    
        clean_doc = ' '.join(txt)
    else:
        clean_doc = doc

    return clean_doc


def lemmatize_doc(doc, stop_list):
    """Lemmatizes document"""
    # Breaking the document to allow consistent lemmatization (see function cleaning())

    doc = nlp(doc)

    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = []
    for token in doc:
        if str(token) in stop_list:  # Add stopword as it is
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


def get_general_word_from_cluster(word_list, wemodel):
    """ Finds the most similar words usind word embedding"""
    try:
        # For W2V
        known_words = [w for w in word_list if w in wemodel.vocab]
    except:
        # For GloVe
        known_words = [w for w in word_list if w in wemodel.index_to_key]
    word_vecs = []
    for w in known_words:
        word_vecs.append(wemodel[w])
    word_vecs = np.array(word_vecs)
    
    # Finding the centorid
    centroid = np.mean(word_vecs, axis=0)

    # print('known words:', len(known_words), 'word_list', len(word_list))
    if len(known_words) > 0:
        # we_word = we_model.most_similar(known_words, topn=1)[0][0]
        we_word = wemodel.most_similar([centroid], topn=1)[0][0]
    else:
        we_word = None
    return we_word


def replace_words_in_df(df_0, cluster_dict, distance_dict, word_dict_copy, col, wemodel, stop_list=None):
    """ Replaces the words in the dataframe """

    out_str = f'Number of clusters: {len(cluster_dict)}   num protected words: {len(stop_list)}'
    logging.debug(out_str)

    # Working with a copy of the df:
    df_copy = df_0

    new_txt = 'general_txt'
    df_copy[new_txt] = df_copy[col].apply(lambda x: lemmatize_doc(x, stop_list=stop_list))

    logging.debug('Going over clusters')

    max_size = 0

    for key, words in cluster_dict.items():
        if key >= 0:  # Ignoring the -1 label
            #if len(cluster_dict[key]) < 20:  # so it will make sense!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Getting the general word
            general_word = get_general_word_from_cluster(words, wemodel).lower()

            if stop_list and (general_word not in stop_list):
                stop_list.append(general_word)  # the list of new words
            rep_str = f'distance: {distance_dict[key]} \tcluster size: {len(words)} \treplacing {words} in {general_word}'
            logging.debug(rep_str)

            if len(words) > max_size: 
                max_size = len(words)

            words_to_replace = []
            for word in words:
                if word not in word_dict_copy:  # Lemmatized word
                    word_dict_copy[word] = {'protected': False, 'lemma': word, 'replaced':False}
                    
                if not word_dict_copy[word]['protected']:
                    # Updaing the word dictionary that the words were replaced
                    word_dict_copy[word]['replaced'] = general_word  # Problem: greate in line 2 miss-identified as replaced
                
                # df_copy[new_txt] = df_copy[new_txt].replace(fr'\b{word}\b', general_word, regex=True)
                
                words_to_replace.append(word)

            # Replacing whole words
            rep_str = create_rep_string(words_to_replace)
            # print('rep_str 1: ', rep_str)
            # rep_str = '\\b|\\b'.join(words_to_replace)
            # print('rep_str 2: ', rep_str)

            df_copy[new_txt] = df_copy[new_txt].replace(fr'\b{rep_str}\b', general_word, regex=True)
    logging.debug('Creating history')
    logging.info(f'Largest clone size - {max_size}')
    df_copy['anon_txt_history'] = df_copy[col].apply(lambda x: print_doc(x, word_dict_copy))

    if stop_list:
        logging.debug('Stop word list was updated')

    return df_copy, word_dict_copy, stop_list


def create_rep_string(word_list):
    """
    Creates a string in the form: 'word1|word2|word3..'
    """
    out_str = f'\\b{word_list[0]}'
    for w in word_list[1:]:
        out_str = f'{out_str}\\b|\\b{w}'
    out_str = f'{out_str}\\b'
    return out_str


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

