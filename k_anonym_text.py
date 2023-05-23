#!/usr/bin/python3

"""
K-anonymity for texts
"""

# Info
__author__ = 'Lior Treiman and Hadas Neuman'
__version__ = '1.1'
__date__ = '23/5/23'

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import argparse
from utils import nlp_utils, cluster_utils, utilization_utils, anonym_utils

def run_anonym(input_file, k, stop_file = None, col='txt'):
    """
    The main function. Runs the anonymization.
    """
    # df from csv
    df = pd.read_csv(input_file)

    # Initilazing the stopword list
    nlp_utils.init_stopwords(stop_file)
    print(f'Stopword list contains {len(nlp_utils.stopword_list)} words')

    # Creating the word dictionary and word list
    word_dict = nlp_utils.create_word_dict(df[col])  # this function takes too long need to make more efficient
    word_list = cluster_utils.get_word_list_for_clustering(word_dict) 

    # Run clustering
    embedded_dict = cluster_utils.embed_corpus(word_list)
    eps_euc = cluster_utils.define_eps_euc()
    cluster_dict, dist_dict, labels = cluster_utils.run_clustering(embedded_dict, cosine=False,eps=eps_euc/2)
    print('Number of clusters:\t', len(cluster_dict))

    # Generalization
    df_2, word_dict_2 = nlp_utils.replace_words_in_df(df, cluster_dict, dist_dict, word_dict)
    curr_k, non_anon_indexes = anonym_utils.get_anonym_degree(docs=df_2[col], min_k=k)
    print(f'Anonymity after generalization: {curr_k}, number of un-anonymized documents: {len(non_anon_indexes)}')

    # Reduction
    force_anon_txt_annoy, neighbor_list = anonym_utils.force_anonym_using_annoy(df_2['anon_txt'], k=k)
    curr_k, non_anon_indexes = anonym_utils.get_anonym_degree(docs=force_anon_txt_annoy, min_k=k)
    print(f'Anonymity after reduction: {curr_k}, number of un-anonymized documents: {len(non_anon_indexes)}')

    # Adding the anonymized corpus to the dataframe
    df_2['force_anon_txt'] = force_anon_txt_annoy
    df_2 = anonym_utils.add_neighbor_list_to_df(df_2, neighbor_list)
    # Counting the number of words and *
    df_2['num_of_words_after_forcing'] = df_2['force_anon_txt'].apply(lambda x: len(re.findall(r'\w+', x)))
    df_2['num_of_deleting_after_forcing'] = df_2['force_anon_txt'].apply(lambda x: len(re.findall(r'\*', x)))

    # Saving
    output_name = f'{os.path.basename(input_file)}_anonym.csv'
    out_file = 'outputs/' + output_name
    df_2.to_csv(out_file, index=False)


def parse_args():
    """
    Defines the ArgumentParser
    :return: The parser
    """
    parser_func = argparse.ArgumentParser(description='Converts dot tree to newick tree format')
    parser_func.add_argument('-f', '--file', help='Input CSV file.', required=True)

    parser_func.add_argument('-k', help='The k anonymity degree',  required=True)
    parser_func.add_argument('-s', '--stop', help='Stop word list file', default=None)
    parser_func.add_argument('--col', help='Text column. Default - txt', default='txt')
    parser_func.add_argument('--silent', action="store_true",
                             help='Prevent the program from displaying screen output. default: False')
    parser_func.add_argument('-version', action='version', version='%(prog)s:' + ' %s-%s' % (__version__, __date__))
    return parser_func


if __name__ == 'main':
    
    print(os.path.basename(__file__), __version__)

    parser = parse_args()
    args = parser.parse_args()

    run_anonym(input_file=args.file, k=args.k, stop_file = args.stop, col=args.col)