# Imports
import pandas as pd
import os
import warnings
import logging
import pkg_resources
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

# from ..utils import nlp_utils
# from ..utils import llm_utils, utilization_utils, anonym_utils
from ..utils import nlp_utils, cluster_utils, utilization_utils, anonym_utils, models

class Kanonym():
    def __init__(self, wemodel:str = 'glove-twitter-25'):
        """
        """
        self.wemodel_name = wemodel
        self.wemodel = models.upload_we_model(wemodel)
        if self.wemodel is None:
            raise Exception(f'Could not upload the word embedding model {self.wemodel_name}') 
        
        if self.wemodel_name == 'glove-twitter-25':
            self.cos = False
        else:
            self.cos = True


    def anonymize(self, df: pd.DataFrame, k: int, col: str = 'txt', plot: bool = False,
                  num_stop: int = 1000, n_jobs: int = 1, verbose: int = 0):
        """
        """
        self._init_logger(verbose)
        logging.info(f'{os.path.basename(__file__)} WE pipeline. WE model: {self.wemodel_name}')

        logging.info(f'Number of documents: {df.shape[0]}')  # Logging

        # Collecting stopword list
        short_stopword_list, long_stopword_list = self._get_stop(num_stop)
        logging.info(f'Stopword list contains {len(long_stopword_list)} words')

        # Creating the word dictionary and word list
        word_dict = nlp_utils.create_word_dict(df[col], long_stopword_list)
        logging.info(f'Number of unique words in dataset: {len(word_dict)}')  # Logging

        # Run clustering
        cluster_dict, dist_dict, _ = cluster_utils.run_clustering(word_dict,
                                                                stop_list=long_stopword_list,
                                                                wemodel=self.wemodel, cosine=self.cos,
                                                                n_jobs=n_jobs)
        logging.info(f'Number of DBSCAN clusters:\t {len(cluster_dict)}')  # Logging

        # Generalization
        df, _, long_stopword_list = nlp_utils.replace_words_in_df(df, cluster_dict,
                                                                dist_dict, word_dict,
                                                                col, wemodel=self.wemodel,
                                                                stop_list=long_stopword_list)
        logging.info(f'Generalization completed.')  # Logging

        # Find k neighbors
        neighbor_list = anonym_utils.ckmeans_clustering(docs=df['general_txt'], k=k, n_jobs=n_jobs,
                                                        stop_list=short_stopword_list)

        logging.info(f'Found {len(neighbor_list)} groups of {k} neighbors')  # Logging

        # Reduction
        force_anon_txt_annoy = anonym_utils.force_anonym(docs=df['general_txt'], neighbor_list=neighbor_list,
                                                     stop_list=long_stopword_list)
        logging.info(f'Reduction completed')  # Logging

        # Testing success
        curr_k, non_anon_indexes = anonym_utils.get_anonym_degree(docs=force_anon_txt_annoy,
                                                                min_k=k, stop_list=long_stopword_list)
        logging.info(f'Anonymity after reduction:{curr_k}  number of un-anonymized documents:' \
                f' {len(non_anon_indexes)}')  # Logging
        
        # Logging the un-anonymized documents
        if len(non_anon_indexes) > 0: 
            logging.info(f'Un-anonymized documents: {non_anon_indexes}')  # Logging
        
        # Adding the anonymized corpus to the dataframe
        anonym_col = 'anon_txt'
        df[anonym_col] = force_anon_txt_annoy
        df = anonym_utils.add_neighbor_list_to_df(df, neighbor_list)

        # Utilization test
        mean_dist = utilization_utils.get_mean_semantice_distance_for_corpus(df[col], df[anonym_col], plot=plot)
        logging.info(f'Mean semantic distance before and after the anonymization process: {mean_dist}')  # Logging
        
        logging.info('Done')  # Logging
        return df, mean_dist
    

    @staticmethod
    def _init_logger(verbose):
        """
        Initiating the logger
        """
        logging.basicConfig(
            level=max(0, 30 - (verbose*10)),
            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    
    
    @staticmethod
    def _get_data_path():
        """
        """
        # Credit: https://medium.com/@vuillaut/python-package-and-access-data-in-your-module-d05e72f58785
        # f1 = pkg_resources.get_resource_filename(__name__, 'data/5000_most_common_words_by_order.txt')
        relative_file = 'data/5000_most_common_words_by_order.txt'
        file_name = pkg_resources.resource_filename('kanonym4text', relative_file)
        return file_name

    
    def _get_stop(self, num_stop):
        """
        Gets the short and the long stop word lists
        """
        # Getting the stopword list
        short_stopword_list = nlp_utils.stopwords.words('english')

        if num_stop < len(short_stopword_list):
            logging.warning(f'The minimal number of stopwords is {len(short_stopword_list)}, based on nltk.corpus.stopwords')
        
        stop_file = self._get_data_path()
        long_stopword_list = nlp_utils.get_list_from_file(stop_file, num_stop)
        
        if long_stopword_list is None:
            raise Exception(f'Could not read the stop word file. {stop_file}')
        
        long_stopword_list = list(set(short_stopword_list + long_stopword_list))

        return short_stopword_list, long_stopword_list
    

    def anonymize_llm(self, df: pd.DataFrame, k: int, col: str = 'txt', plot: bool = False,
                  n_jobs: int = 1, verbose: int = 0):
        """
        Parameters
        ----------
        df - dataframe recieved from the user containing the original corpus (list of documents)
        k - number of times each document appears in a data set (required level of k-anonymity)
        col - colom name in the df contains the documents (default: 'txt')
        plot - True if user want's to get visuales during the process (default: False)
        n_jobs - The number of parallel jobs to run for neighbors search (default: -1, means all processors)
        verbose - The verbosity level: if non zero, progress messages are printed (default: 0)

        Returns
        -------
        Anonymized dataframe
        Average semantic distance
        """

        from ..utils import llm_utils

        self._init_logger(verbose)
        logging.info(f'{os.path.basename(__file__)} LLM pipeline')

        # Adding number of characters
        num_chars_col = 'Num_characters'
        df[num_chars_col] = df[col].str.len()
        docs = df[col]
        
        logging.info(f'Number of documents: {len(docs)}')
        logging.info(f'Average number of characters in documents: {df[num_chars_col].mean()} '
                    f'maximum characters in a document {df[num_chars_col].max()}')
        
        # Removing the Num_characters column
        df.drop([num_chars_col], axis=1, inplace=True)

        # Running the anonymization
        annon_docs, _ = llm_utils.run_anonymization_on_txt(docs, k, n_jobs)
        anonum_col = 'llm_anonym_text'
        df[anonum_col] = annon_docs

        curr_k, non_anon_indexes = anonym_utils.get_anonym_degree(docs=annon_docs, min_k=k)
        logging.info(f'Anonymity after reduction:{curr_k}  number of un-anonymized documents: {len(non_anon_indexes)}')  # Logging

        # Logging the un-anonymized documents
        if len(non_anon_indexes) > 0: 
            logging.info(f'Un-anonymized documents: {non_anon_indexes}')  # Logging

        # Utilization test
        mean_dist = utilization_utils.get_mean_semantice_distance_for_corpus(df[col], df[anonum_col],
                                                                            plot=plot)
        logging.info(f'Mean semantic distance before and after the anonymization process: {mean_dist}')  # Logging

        logging.info('Done')  # Logging
        return df, mean_dist

        







