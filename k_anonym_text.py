#!/usr/bin/python3

"""
K-anonymity for texts
"""

# Info
__author__ = 'Lior Treiman and Hadas Neuman'
__version__ = '1.1'
__date__ = '3/6/23'

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
from datetime import date
import logging
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


def llm_method(arguments):
    """
    Uses LLM methods to preform anonymization
    """
    from utils import llm_utils, utilization_utils, anonym_utils

    # getting the input arguments
    input_file = arguments.file  # Input database
    k = int(arguments.k)  # Desired k degree
    col = arguments.col  # The text columns
    prefix = get_prefix(arguments)
    n_jobs = arguments.n_jobs
    plot = arguments.plot

    # Creating the datafrmae
    df = pd.read_csv(input_file)

    # Adding number of characters
    num_chars_col = 'Num_characters'
    df[num_chars_col] = df[col].str.len()
    docs = df[col]
    
    logging.info(f'Number of documents: {len(docs)}')
    logging.info(f'Average number of characters in documents: {df[num_chars_col].mean()} maximum characters in a document {df[num_chars_col].max()}')

    # Runing the anonymization
    annon_docs, _ = llm_utils.run_anonymization_on_txt(docs, k, n_jobs)
    df['anonymized_text'] = annon_docs

    curr_k, non_anon_indexes = anonym_utils.get_anonym_degree(docs=annon_docs, min_k=k)
    out_str = f'Anonymity degree:\t\t{curr_k}\t number of un-anonymized documents: \t{len(non_anon_indexes)}'
    logging.info(out_str)  # Logging

    # Logging the un-anonymized documents
    if len(non_anon_indexes) > 0: 
        out_str = f'Un-anonymized documents: {non_anon_indexes}'
        logging.info(out_str)  # Logging

    # Utilization utils
    if plot:
        plot_prefix = prefix
    else:
        plot_prefix = None
    mean_dist = utilization_utils.get_mean_semantice_distance_for_corpus(df[col], df['anonymized_text'], prefix=plot_prefix)
    out_str = f'Mean semantic distance before and after the anonymization process: {mean_dist}'
    logging.info(out_str)  # Logging

    # Saving
    if arguments.out:
        output_name = arguments.out
    else:
        output_name = f'{prefix}_anonymized.csv'
    out_file = 'outputs/' + output_name
    df.to_csv(out_file, index=False)

    out_str = f'Done. Output saved to {out_file}'
    logging.info(out_str)  # Logging


def get_prefix(arguments):
    """
    Creates a prefix based on the input file and k.
    """
    # Removing extension
    if arguments.out:
        file = arguments.out
    else:
        file = arguments.file

    _, file_extension = os.path.splitext(file)
    base_name = os.path.basename(file).replace(file_extension, '')
    prefix = f'{base_name}_k{arguments.k}'

    # LLM
    if arguments.llm:
        prefix += '_llm'

    return prefix


def init_logger(verbose):
    """
    Initiating the logger
    """
    # log_name = f'logs/{prefix}.log'
    logging.basicConfig(
    level=max(0, 30 - (verbose*10)),
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')


# def run_anonym(arguments):
def run_anonym(df: pd.DataFrame, k: int, col: str='txt', plot: bool=False, n_jobs: int = 1, verbose: int=0):
    """
    The main function. Runs the anonymization.
    """
    from utils import nlp_utils, cluster_utils, utilization_utils, anonym_utils

    # # getting the input arguments
    # input_file = arguments.file  # Input database
    # k = int(arguments.k)  # Desired k degree
    # stop_file = arguments.stop  # File with list of stop words
    # col = arguments.col  # The text columns  
    # n_jobs = args.n_jobs
    # plot = args.plot

    # # df from csv
    # df = pd.read_csv(input_file)

    # prefix = get_prefix(args) 

    # TEMP
    prefix = 'temp_prefix' # TEMP
    stop_file = 'data/1000_most_common_words.txt'
    init_logger(verbose)
    logging.info('Start')  # Logging
    
    nlp_utils.short_stopword_list = nlp_utils.stopwords.words('english')
    nlp_utils.long_stopword_list = list(set(nlp_utils.short_stopword_list + nlp_utils.get_list_from_file(stop_file)))

    # out_str = f'Stopword list contains {len(nlp_utils.stopword_list)} words'
    out_str = f'Stopword list contains {len(nlp_utils.short_stopword_list)}, {len(nlp_utils.long_stopword_list)} words'
    logging.info(out_str)  # Logging

    # Creating the word dictionary and word list
    word_dict = nlp_utils.create_word_dict(df[col], nlp_utils.long_stopword_list)  # this function takes too long need to make more efficient
    out_str = f'Number of unique words in dataset: {len(word_dict)}'
    logging.info(out_str)  # Logging

    # Run clustering
    cluster_dict, dist_dict, _ = cluster_utils.run_clustering(word_dict, stop_list=nlp_utils.long_stopword_list, cosine=True, n_jobs=n_jobs)
    out_str = f'Number of DBSCAN clusters:\t {len(cluster_dict)}'
    logging.info(out_str)  # Logging

    # Generalization
    df, _ = nlp_utils.replace_words_in_df(df, cluster_dict, dist_dict, word_dict, prefix=prefix)
    out_str = f'Generalization completed.'
    logging.info(out_str)  # Logging
    
    # Test current k - a waist of time
    # logging.info('Testing anonymity...')  # Logging
    # curr_k, non_anon_indexes = anonym_utils.get_anonym_degree(docs=df[col], min_k=k)
    # out_str = f'Anonymity after generalization:\t{curr_k}\t number of un-anonymized documents: \t{len(non_anon_indexes)}'
    # logging.info(out_str)  # Logging

    # Find k neighbors
    #neighbor_list = anonym_utils.find_k_neighbors_using_annoy(docs=df['anon_txt'], k=k)
    neighbor_list = anonym_utils.ckmeans_clustering(docs=df['anon_txt'], k=k, n_jobs=n_jobs)

    out_str = f'Found {len(neighbor_list)} groups of {k} neighbors'
    logging.info(out_str)  # Logging
    
    # Reduction
    force_anon_txt_annoy = anonym_utils.force_anonym(docs=df['anon_txt'], neighbor_list=neighbor_list)
    curr_k, non_anon_indexes = anonym_utils.get_anonym_degree(docs=force_anon_txt_annoy, min_k=k)
    out_str = f'Anonymity after reduction:\t\t{curr_k}\t number of un-anonymized documents: \t{len(non_anon_indexes)}'
    logging.info(out_str)  # Logging

    # Logging the un-anonymized documents
    if len(non_anon_indexes) > 0: 
        out_str = f'Un-anonymized documents: {non_anon_indexes}'
        logging.info(out_str)  # Logging

    # Adding the anonymized corpus to the dataframe
    anonym_col = 'force_anon_txt'
    df[anonym_col] = force_anon_txt_annoy
    del(force_anon_txt_annoy)  # Freeing space
    df = anonym_utils.add_neighbor_list_to_df(df, neighbor_list)
    # Counting the number of words and *
    df['num_of_words_after_forcing'] = df['force_anon_txt'].apply(lambda x: len(re.findall(r'\w+', x)))
    df['num_of_deleting_after_forcing'] = df['force_anon_txt'].apply(lambda x: len(re.findall(r'\*', x)))

    # Utilization utils
    if plot:
        plot_prefix = prefix
    else:
        plot_prefix = None
    mean_dist = utilization_utils.get_mean_semantice_distance_for_corpus(df[col], df[anonym_col], prefix=plot_prefix)
    out_str = f'Mean semantic distance before and after the anonymization process: {mean_dist}'
    logging.info(out_str)  # Logging

    # # Saving
    # if arguments.out:
    #     output_name = arguments.out
    # else:
    #     output_name = f'{prefix}_anonymized.csv'
    # out_file = 'outputs/' + output_name
    # df.to_csv(out_file, index=False)

    # out_str = f'Done. Output saved to {out_file}'
    
    logging.info('Done')  # Logging
    return df


def parse_args():
    """
    Defines the ArgumentParser
    :return: The parser
    """
    parser_func = argparse.ArgumentParser(description='Converts dot tree to newick tree format')
    parser_func.add_argument('-f', '--file', help='Input CSV file', required=True)

    parser_func.add_argument('-k', help='The k anonymity degree',  required=True)
    parser_func.add_argument('-s', '--stop', help='Stop word list file. default=data/1000_most_common_words.txt', 
                             default='data/1000_most_common_words.txt')
    parser_func.add_argument('--col', help='Text column. Default - txt', default='txt')
    parser_func.add_argument('--out', help='Output file name. default - based on input file and k')
    parser_func.add_argument('--verbose', type=int, default=0,
                             help='Prevent the program from displaying screen output. default: 0')
    parser_func.add_argument('--llm', action="store_true",
                             help='Use LLM methods. default: False')
    parser_func.add_argument('--n_jobs',
                             help='The number of parallel jobs to run. -1 means using all processors. default -1', 
                             type=int, default=-1)
    parser_func.add_argument('--plot', action="store_true",
                             help='Plot semantic distance before and after the anonymization. default: False')
    parser_func.add_argument('-version', action='version', version='%(prog)s:' + ' %s-%s' % (__version__, __date__))
    return parser_func


#if __name__ == 'main':
if False:
    print(os.path.basename(__file__), __version__)
    
    start_time = time.time()

    parser = parse_args()
    args = parser.parse_args()

    prefix = get_prefix(args)

    # Initiating the logger
    log_name = f'logs/{prefix}.log'
    logging.basicConfig(
    filename=log_name,
    level=max(0, 30 - (args.verbose*10)),
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    logging.info(f'Starting.')
    logging.info(f'Input arguments = {args}')

    if not args.llm:
        run_anonym(args)
    else:
        llm_method(args)
    
    logging.info(f'Running time: {round((time.time() - start_time),2)} seconds')
    
    