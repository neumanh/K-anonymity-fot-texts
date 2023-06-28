from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from . import anonym_utils


def print_example(indexes, origina_docs, new_docs):
    print('Before:')
    for i in indexes:
        print(origina_docs[i])
    print('\nAfter:')
    for i in indexes:
        print(new_docs[i])


def sum_text_0(doc_list, tokenizer, gen_model):
    # define the input sentences
    #input_text = '. '.join(doc_list)
    input_text = ''
    for doc in doc_list:
       input_text = f'{input_text}\n- {doc}. ' 

    # preprocess the input sentences
    input_ids = tokenizer.encode(f'Generate a general form of these sentences: \n{input_text}', return_tensors="pt")

    # generate the summary sentence
    output_ids = gen_model.generate(input_ids=input_ids, max_length=32, num_beams=4, early_stopping=True)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output


def sum_text(doc_list, tokenizer, gen_model):
    # define the input sentences
    #input_text = '. '.join(doc_list)
    input_text = ''
    i = 1
    for doc in doc_list:
       input_text = f'{input_text}{i}: {doc}. ' 
       i += 1
    
    # print('doc_list:', doc_list)
    # preprocess the input sentences
    prompt = prompt_builder(doc_list)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", show_progress_bar=False)

    # generate the summary sentence
    output_ids = gen_model.generate(input_ids=input_ids, max_length=32, num_beams=4, early_stopping=True)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output


def run_anonymization_on_txt(docs, k, n_jobs):
    """ Finding K nearest neighbors and summarize them """
    
    # Defining models
    # google/flan-t5-xl
    # "google/flan-t5-xl"
    tok_model_name = 'google/flan-t5-xl'
    gen_model_name = 'google/flan-t5-xl'
    emb_model_name = 'sentence-transformers/all-MiniLM-L12-v1'

    logging.debug(f'Tokenizer: {tok_model_name}  Genrative model: {gen_model_name}  Embedding model: {emb_model_name}')
    
    # Uploading models
    tokenizer = AutoTokenizer.from_pretrained(tok_model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
    emb_model = SentenceTransformer(emb_model_name)

    annon_docs = docs.copy()
    # Embedding
    logging.info('Embedding the documents...')
    docs_emb = emb_model.encode(docs, show_progress_bar=False)
    logging.debug(f'Embedding dimensions: {docs_emb.shape}, {type(docs_emb)}')
    # neighbor_list = ckmeans_clustering(docs_emb, k)
    neighbor_list = anonym_utils.ckmeans_clustering(docs_emb, k=k, n_jobs=1, dim_reduct=False)
    logging.info(f'Found {len(neighbor_list)} groups of {k} neighbors')
    # print('&&&&&&&&&&&', 3)

    logging.info(f'Generating alternative documents...')
    for n in neighbor_list:
        # print('n:', n)
        curr_docs = []
        for doc_index in n:
            # Adding the document to the similar doc list
            curr_docs.append(docs[doc_index])
        sum_doc = sum_text(curr_docs, tokenizer, gen_model)
        # sum_doc = sum_text_0(curr_docs, tokenizer, gen_model)
        # print('similar_doc_ind', n, '\tSummary:', sum_doc)
        for doc_index in n:
            annon_docs[doc_index] = sum_doc
    # print('&&&&&&&&&&&', 4)
    
    logging.info(f'Generation completed.')

    return annon_docs, neighbor_list


def prompt_builder(docs):
    """
    Generating the prompt for document generalization
    """

    # Removing /n from documens
    new_docs = []
    for d in docs:
        new_docs.append(d.replace('\n',' '))
    
    # Adding new line and '-' between documnets
    sents = '\n-'.join(new_docs)
    # Adding '-' before the first document
    sents = '-' + sents + '\n'

    prompt = f'''Write a general sentence that best represents each of the following sentences and is true to all of them. For example: 

Sentences:
-the alchemist: sure this is an interesting book. unfortunately the copy we received was written in spanish!
-immensee: a beautiful story, but the translation from the original language (german) is unbelievably poor.
Result: 
Good book, problem with the translation.

Sentences:
-low budget: some good action and scenery, but otherwise pretty low budget movie. only recommended for die-hard horror fans.
-horrible movie!: please listen everybody this movie is terrible-don\'t bother ok!it is just dumb not scary!
Result: 
The movie was bad.

Sentences:
-america the beautiful: this book works great for teaching the song. the hardback cover is great for multiple uses.
-enjoyable music: bought this to use for our church\'s hawaiaan-harvest festival for children ages 4-12. great music!
Result: 
Good and useful music.

Sentences:
-invaluable: the authors have provided the quintessential study guide to the canterbury tales. this book is invaluable.
-magical: wonderful characters, stories within stories. this book offers beautiful writing, a deep spirituality and a happy ending!
Result: 
Wonderful book and great writing.

Sentences:
-textbook: book shipped quickly and was in excellent condition as stated. easy transaction would buy again
-mathbook: excellent condition of book. description is true to the condition of the mathbook.shipped quickly excellent seller.
-good deal!: used library book, but still in good condition. book came quickly and was cheap. overall good deal!
Result: 
Good condition book, good delivery and good seller.

Sentences:
-great: wishmaster is a fantastic album, that should grace everyone's music collection. innovating and always refreshing. truly a masterpiece.
-fantastic debut: a great debut record from a band who deserve a lot of attention. buy this record twice.
-great music, great arrangements and performance.: very satisfied with this cd. cem duruoz is a genius.
Result: 
Great music, highly recommended.

Sentences:
{sents}
Result:
'''
    return prompt


