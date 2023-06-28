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
    from utils import llm_utils, utilization_utils

    # getting the input arguments
    input_file = arguments.file  # Input database
    k = int(arguments.k)  # Desired k degree
    col = arguments.col  # The text columns
    prefix = get_prefix(arguments) + '_llm'
    out_str = f'Starting. input_file={input_file}  k={k}  col={col} llm={arguments.llm}'
    logging.info(out_str)  # Logging

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
    prefix = f'{base_name}_{arguments.k}'
    if arguments.stop:
        _, file_extension = os.path.splitext(arguments.stop)
        base_stop_name = os.path.basename(arguments.stop).replace(file_extension, '')
        prefix += f'_stop_{base_stop_name}'
    return prefix


def log(message, log_file = None):
    """
    Writes the progress to the log file
    """
    pass
    # if not log_file:
    #     log_file = f'logs/{prefix}.log'
    # out_str = f'{prefix}\t{date.today()}\t{datetime.now().strftime("%H:%M:%S")}\t{message}'
    # with open(log_file, 'a') as f:
    #     f.write(out_str)
    #     f.write('\n')

def init_logger(verbose):
    """
    Initiating the logger
    """
    # log_name = f'logs/{prefix}.log'
    logging.basicConfig(
        filename="logs/test_clustering.log",
        level=0,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')


def run_anonym(arguments):
    """
    The main function. Runs the anonymization.
    """
    from utils import nlp_utils, cluster_utils, utilization_utils, anonym_utils, models

    df = pd.read_csv(arguments.file)
    k = int(arguments.k)

    init_logger(1)
    col='txt'
    wemodel = 'word2vec-google-news-300'
    n_jobs = -1

    logging.info('Start')  # Logging
    logging.info(f'Word embedding model: {wemodel}')  # Logging

    # Uploading the word embedding model
    if wemodel == 'glove-twitter-25':
        cos = False
    else:
        cos = True
    
    wemodel = models.upload_we_model(wemodel)
    if wemodel is None:
        exit(1) 

    # TEMP
    prefix = 'temp_prefix' # TEMP
    stop_file = 'data/1000_most_common_words.txt'

    logging.info(f'Number of documents: {df.shape[0]}')  # Logging
    
    nlp_utils.short_stopword_list = nlp_utils.stopwords.words('english')
    nlp_utils.long_stopword_list = list(set(nlp_utils.short_stopword_list + nlp_utils.get_list_from_file(stop_file)))

    out_str = f'Stopword list contains {len(nlp_utils.long_stopword_list)} words'
    logging.info(out_str)  # Logging

    # Creating the word dictionary and word list
    word_dict = nlp_utils.create_word_dict(df[col], nlp_utils.long_stopword_list)  # this function takes too long need to make more efficient
    out_str = f'Number of unique words in dataset: {len(word_dict)}'
    logging.info(out_str)  # Logging

    # Run clustering
    cluster_dict, dist_dict, _ = cluster_utils.run_clustering(word_dict, stop_list=nlp_utils.long_stopword_list, wemodel=wemodel, cosine=cos, n_jobs=n_jobs)
    out_str = f'Number of DBSCAN clusters:\t {len(cluster_dict)}'
    logging.info(out_str)  # Logging

    # Generalization
    df, _ = nlp_utils.replace_words_in_df(df, cluster_dict, dist_dict, word_dict, col, wemodel=wemodel)
    del(wemodel)  # Freeing space
    out_str = f'Generalization completed.'
    logging.info(out_str)  # Logging
    
    # Find k neighbors
    # force_anon_txt_annoy, neighbor_list = anonym_utils.force_anonym_using_annoy(df['anon_txt'], k=k)
    # neighbor_list = anonym_utils.find_k_neighbors_using_annoy(docs=df['anon_txt'], k=k)
    # for n in [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000]:
    for d in [10, 20, 30, 50, 70, 100]:
        n=df.shape[0]
        # df_temp = df.head(n=n)
        df_temp = df
        logging.info('Start clustering...')  # Logging 
        neighbor_list = anonym_utils.ckmeans_clustering(docs=df_temp['anon_txt'], k=k, dim_reduct=d)
        logging.info('create_neighbors_df_for_comparison...')  # Logging
        temp_df = anonym_utils.create_neighbors_df_for_comparison(df_temp['anon_txt'], neighbor_list)
        temp_prefix = f'testing_ckmeans_{n}_{prefix}'
        temp_file = f'temp/{temp_prefix}.csv'
        temp_df.to_csv(temp_file, index=False)
        logging.info('Calculating distance...')  # Logging
        mean_dist = utilization_utils.get_mean_semantice_distance_for_corpus(temp_df[0], temp_df[1], prefix=temp_prefix)
        # logging.info(f'Mean sentiment distance: {utilization_utils.get_mean_semantice_distance_for_corpus(temp_df[0], temp_df[1], prefix=temp_prefix)}')  # Logging
        out_str = f'ckmeans\td={d}\t n={n} Found {len(neighbor_list)} groups of {k} neighbors. mean_dist: {mean_dist}'
        logging.info(out_str)  # Logging
    
    # for n in [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000]:
    #     df_temp = df.head(n=n)
    #     neighbor_list = anonym_utils.find_k_neighbors_using_annoy(docs=df_temp['anon_txt'], k=k)
    #     out_str = f'Annoy\tn={n}\tFound {len(neighbor_list)} groups of {k} neighbors'
    #     logging.info(out_str)  # Logging
    #     temp_df = anonym_utils.create_neighbors_df_for_comparison(df_temp['anon_txt'], neighbor_list)
    #     temp_prefix = f'testing_annoy_{n}_full_vecs'
    #     temp_file = f'temp/{temp_prefix}.csv'
    #     temp_df.to_csv(temp_file, index=False)
    #     logging.info(f'Mean sentiment distance: {utilization_utils.get_mean_semantice_distance_for_corpus(temp_df[0], temp_df[1], prefix=temp_prefix)}')  # Logging

    # # Reduction
    # force_anon_txt_annoy = anonym_utils.force_anonym(docs=df['anon_txt'], k=k, neighbor_list=neighbor_list)
    # curr_k, non_anon_indexes = anonym_utils.get_anonym_degree(docs=force_anon_txt_annoy, min_k=k)
    # out_str = f'Anonymity after reduction:\t\t{curr_k}\t number of un-anonymized documents: \t{len(non_anon_indexes)}'
    # logging.info(out_str)  # Logging

    # # Logging the un-anonymized documents
    # if len(non_anon_indexes) > 0: 
    #     out_str = f'Un-anonymized documents: {non_anon_indexes}'
    #     logging.info(out_str)  # Logging

    # # Adding the anonymized corpus to the dataframe
    # anonym_col = 'force_anon_txt'
    # df[anonym_col] = force_anon_txt_annoy
    # del(force_anon_txt_annoy)  # Freeing space
    # df = anonym_utils.add_neighbor_list_to_df(df, neighbor_list)
    # # Counting the number of words and *
    # df['num_of_words_after_forcing'] = df['force_anon_txt'].apply(lambda x: len(re.findall(r'\w+', x)))
    # df['num_of_deleting_after_forcing'] = df['force_anon_txt'].apply(lambda x: len(re.findall(r'\*', x)))

    # # Utilization utils
    # mean_dist = utilization_utils.get_mean_semantice_distance_for_corpus(df[col], df[anonym_col], prefix=prefix)
    # out_str = f'Mean semantic distance before and after the anonymization process: {mean_dist}'
    # logging.info(out_str)  # Logging

    # # Saving
    # if arguments.out:
    #     output_name = arguments.out
    # else:
    #     output_name = f'{prefix}_anonymized.csv'
    # out_file = 'outputs/' + output_name
    # df.to_csv(out_file, index=False)

    # out_str = f'Done. Output saved to {out_file}'
    # logging.info(out_str)  # Logging


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

    prefix = get_prefix(args)

    # Initiating the logger
    log_name = f'logs/{prefix}.log'
    logging.basicConfig(
    filename=log_name,
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    logging.info(f'Started. Arguments = {args}')

    if not args.llm:
        run_anonym(args)
    else:
        llm_method(args)
    
    print(f'Running time: {round((time.time() - start_time),2)} seconds')
    
    # current_time = datetime.now().strftime("%H:%M:%S")
    # print("End Time =", current_time)

    