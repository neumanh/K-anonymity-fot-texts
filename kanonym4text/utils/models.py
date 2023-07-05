import gensim.downloader as api
import nltk
import logging

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def upload_we_model(model_name):
    # From GloVe
    # model_name = 'glove-twitter-25'

    possible_models = list(api.info()['models'].keys())
    if model_name not in possible_models:
        logging.error(f'The model {model_name} is not known to Gensim')
    # print('possible models:', possible_models)

    try:
        model = api.load(model_name)

    except Exception as e:
        logging.error(f'Could not upload the model {model_name}', e)
        model = None
    
    return model

we_model = upload_we_model(model_name='word2vec-google-news-300')