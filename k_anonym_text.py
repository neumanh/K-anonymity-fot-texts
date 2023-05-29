#!/usr/bin/python3

"""
K-anonymity for texts
"""

# Info
__author__ = 'Lior Treiman and Hadas Neuman'
__version__ = '1.1'
__date__ = '27/5/23'

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import argparse
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


def llm_method(arguments):
    """
    Uses LLM methods to preform anonymization
    """
    from utils import llm_utils, utilization_utils

    # getting the input arguments
    input_file = arguments.file  # Input database
    k = int(arguments.k)  # Desired k degree
    col = arguments.col  # The text columns
    prefix = get_prefix(input_file, k) + '_llm'
    out_str = f'Starting. input_file={input_file}  k={k}  col={col} llm={arguments.llm}'
    log(prefix, out_str)  # Logging

    # Creating the datafrmae
    df = pd.read_csv(input_file)
    docs = df[col]

    # Runing the anonymization
    annon_docs, _ = llm_utils.run_anonymization_on_txt(docs, k)
    print(len(docs))
    print(len(annon_docs))
    # print(annon_docs)
    df['anonymized_text'] = annon_docs

    print(df[col])
    print(df['anonymized_text'])


    # Utilization utils
    mean_dist = utilization_utils.get_mean_semantice_distance_for_corpus(df[col], df['anonymized_text'], prefix=prefix)
    out_str = f'Mean semantic distance before and after the anonymization process: {mean_dist}'
    log(prefix, out_str)  # Logging

    # Saving
    if arguments.out:
        output_name = arguments.out
    else:
        output_name = f'{prefix}_anonymized.csv'
    out_file = 'outputs/' + output_name
    df.to_csv(out_file, index=False)

    out_str = f'Done. Output saved to {out_file}'
    log(prefix, out_str)  # Logging


def get_prefix(input_file, k):
    """
    Creates a prefix based on the input file and k.
    """
    _, file_extension = os.path.splitext(input_file)
    # Removing extension
    base_name = os.path.basename(input_file).replace(file_extension, '')
    prefix = f'{base_name}_{k}'
    return prefix


def log(prefix, message, log_file = None):
    """
    Writes the progress to the log file
    """
    if not log_file:
        log_file = f'{prefix}.log'
    out_str = f'{prefix}\t{datetime.now().strftime("%H:%M:%S")}\t{message}'
    with open(log_file, 'a') as f:
        f.write(out_str)
        f.write('\n')


def run_anonym(arguments):
    """
    The main function. Runs the anonymization.
    """
    from utils import nlp_utils, cluster_utils, utilization_utils, anonym_utils

    # getting the input arguments
    input_file = arguments.file  # Input database
    k = int(arguments.k)  # Desired k degree
    stop_file = arguments.stop  # File with list of stop words
    col = arguments.col  # The text columns   

    prefix = get_prefix(input_file, k)
    out_str = f'Starting. input_file={input_file}  k={k}  stop_file={stop_file} col={col}'
    log(prefix, out_str)  # Logging

    # df from csv
    df = pd.read_csv(input_file)

    # Initilazing the stopword list
    nlp_utils.init_stopwords(stop_file)
    out_str = f'Stopword list contains {len(nlp_utils.stopword_list)} words'
    log(prefix, out_str)  # Logging

    # Creating the word dictionary and word list
    word_dict = nlp_utils.create_word_dict(df[col])  # this function takes too long need to make more efficient

    # Run clustering
    cluster_dict, dist_dict, _ = cluster_utils.run_clustering(word_dict, cosine=False)
    out_str = f'Number of clusters:\t {len(cluster_dict)}'
    log(prefix, out_str)  # Logging

    # Generalization
    df, _ = nlp_utils.replace_words_in_df(df, cluster_dict, dist_dict, word_dict)
    curr_k, non_anon_indexes = anonym_utils.get_anonym_degree(docs=df[col], min_k=k)
    out_str = f'Anonymity after generalization:\t{curr_k}\t number of un-anonymized documents: \t{len(non_anon_indexes)}'
    log(prefix, out_str)  # Logging

    # Reduction
    force_anon_txt_annoy, neighbor_list = anonym_utils.force_anonym_using_annoy(df['anon_txt'], k=k)
    curr_k, non_anon_indexes = anonym_utils.get_anonym_degree(docs=force_anon_txt_annoy, min_k=k)
    out_str = f'Anonymity after reduction:\t\t{curr_k}\t number of un-anonymized documents: \t{len(non_anon_indexes)}'
    log(prefix, out_str)  # Logging

    # Logging the un-anonymized documents
    if len(non_anon_indexes) > 0: 
        out_str = f'Un-anonymized documents: {non_anon_indexes}'
        log(prefix, out_str)  # Logging

    # Adding the anonymized corpus to the dataframe
    anonym_col = 'force_anon_txt'
    df[anonym_col] = force_anon_txt_annoy
    del(force_anon_txt_annoy)  # Freeing space
    df = anonym_utils.add_neighbor_list_to_df(df, neighbor_list)
    # Counting the number of words and *
    df['num_of_words_after_forcing'] = df['force_anon_txt'].apply(lambda x: len(re.findall(r'\w+', x)))
    df['num_of_deleting_after_forcing'] = df['force_anon_txt'].apply(lambda x: len(re.findall(r'\*', x)))

    # Utilization utils
    mean_dist = utilization_utils.get_mean_semantice_distance_for_corpus(df[col], df[anonym_col], prefix=prefix)
    out_str = f'Mean semantic distance before and after the anonymization process: {mean_dist}'
    log(prefix, out_str)  # Logging

    # Saving
    if arguments.out:
        output_name = arguments.out
    else:
        output_name = f'{prefix}_anonymized.csv'
    out_file = 'outputs/' + output_name
    df.to_csv(out_file, index=False)

    out_str = f'Done. Output saved to {out_file}'
    log(prefix, out_str)  # Logging


def parse_args():
    """
    Defines the ArgumentParser
    :return: The parser
    """
    parser_func = argparse.ArgumentParser(description='Converts dot tree to newick tree format')
    parser_func.add_argument('-f', '--file', help='Input CSV file', required=True)

    parser_func.add_argument('-k', help='The k anonymity degree',  required=True)
    parser_func.add_argument('-s', '--stop', help='Stop word list file', default=None)
    parser_func.add_argument('--col', help='Text column. Default - txt', default='txt')
    parser_func.add_argument('--out', help='Output file name. default - based on input file and k')
    parser_func.add_argument('--silent', action="store_true",
                             help='Prevent the program from displaying screen output. default: False')
    parser_func.add_argument('--llm', action="store_true",
                             help='Use LLM methods. default: False')
    parser_func.add_argument('-version', action='version', version='%(prog)s:' + ' %s-%s' % (__version__, __date__))
    return parser_func


#if __name__ == 'main':
if True:
    print(os.path.basename(__file__), __version__)
    
    start_time = time.time()

    parser = parse_args()
    args = parser.parse_args()

    if not args.llm:
        run_anonym(args)
    else:
        llm_method(args)
    
    print(f'Running time: {round((time.time() - start_time),2)} seconds')
    
    # current_time = datetime.now().strftime("%H:%M:%S")
    # print("End Time =", current_time)