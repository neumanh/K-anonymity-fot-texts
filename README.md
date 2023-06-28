# Kanon4txt K-ANONYMITY GUARANTEE FOR TEXTS



K Anonymity for text (or in short - Kanon4txt) is an open‑sourced library that receives a corpus from the user and returns a K anonymized corpus with additional info about the anonymization processing performed.
Kanon4txt is designed to be easily utilized, to guarantee anonymization at a certain level pre-defined by the user (k) while still preserving some of the text utilization properties. 
This repo and package are part of our Y-Data data science final project, and we would love to hear your feedback and learn from it!

## Overview - TBD
In this project, we aim to apply data science techniques to anonymize textual data while preserving their utility. K-Anonymity is a technique used to ensure that an individual in a dataset cannot be identified by linking their attributes to external information, by forcing each row to be identical to k-1 other rows The anonymized data can be used for various purposes like data sharing, research, and analysis without compromising privacy. We plan on creating a novel algorithm for k-anonymity. Specifically, we address the case of unstructured data items, such as texts. Using various NLP techniques, from classical to modern DL-based solutions, and testing the utility of the anonymized data.
We have tested the library on two main datasets:

## Guiding Principles (Algorithm) - TBD
Weighted KNN for efficient large-scale search
PCA for reducing the dimensionality of the search space
Binary search for optimizing the counterfactuals
Multiprocessing for parallelizing the search and utilizing the full power of the machine

Data Preprocessing: tokenization, stemming, and stop word removal.
K-Anonymization: Generalization and Reduction (see more details on algorithms at TBD).
Evaluation: The effectiveness is evaluated by embedding distance and semantic score.
Visualization: dataset properties prior to and post anonymization will be visualized.

Amazon dataset,
Enron emails dataset
and we show it is able to generate anonymized corpora in both cases.


# Getting Started
You can get started with Kanon4txt immediately by installing it with pip:
## download package
pip install Kanon4txt

## step 1 - creat a data frame from your corpus
The code receives a data frame containing the corpus in the following format:
"txt" - column with the texts

## step 2 - import the model:

from Kanon4txt.nlp_utils import txt_pre_process
TBD
TBD

## Step 3 - create a config for the anonymizer:

config = {"k": int } (k can be any number between 1-5)
              
## Step 4 -  run the anonymization process on your corpus


df_output = RunKanonym(df,config,verbos=0)  (RunKanonym ==run_anonym) to be updated incode



# RunKanonym (add the name of functions for each step)
The RunKanonym has the following stages:

### Pre-processing (on the entire dataset) -


### Clustering the data using DBSCAN based on the PCA components


### Generalizing

### Reduction

### Evaluation
We check if the all dataset is k-anonymized.by checking if any of the documents doesn't have k-1 similar documents 
If it does not we mark it as invalid



## Parameters:

df - the DataFrame to use with the entire dataset you want to use for anonymization guarantee
model - the embedding model to use for word embedding for generalization step 
config - the config dictionary


n_jobs - the number of CPU cores to use for multiprocessing, use -1 for all available CPU cores and 1 for no multiprocessing
verbose - whether to print extra information - useful for debugging, the default is 0 (no printing)


# Support

## Create a Bug Report
If you see an error message or run into an issue, please create a bug report. This effort is valued and helps all users.

## Submit a Feature Request
If you have an idea, or you're missing a capability that would make development easier and more robust, please Submit a feature request.

If a similar feature request already exists, don't forget to leave a "+1". If you add some more information such as your thoughts and vision about the feature, your comments will be embraced warmly :)

# Contributing

Kanon4txt is an open-source project. We are committed to a fully transparent development process and highly appreciate any contributions. Whether you are helping us fix bugs, proposing new features, improving our documentation, or spreading the word - we would love to have you!




