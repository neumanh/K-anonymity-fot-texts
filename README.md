# kanonym4text K-anonymity guerantee for texts



K Anonymity for text (or in short - kanonym4text) is an open‑sourced library that receives a corpus from the user (*dataframe*) and returns a K-anonymized corpus with additional information about the anonymization process performed.
kanon4txet is designed to be easily utilized, to **guarantee** anonymization at a certain level pre-defined by the user (k) while still preserving some of the text utilization properties. 
This repo and package are part of our Y-Data data science final project, and we would love to hear your feedback and learn from it!

## Overview
In this project, we aim to apply data science techniques to anonymize textual data while preserving their utility. K-Anonymity is a technique used to ensure that an individual in a dataset cannot be identified by linking their attributes to external information, by forcing each row to be identical to k-1 other rows The anonymized data can be used for various purposes like data sharing, research, and analysis without compromising privacy. We plan on creating a novel algorithm for k-anonymity. Specifically, we address the case of unstructured data items, such as texts. Using various NLP techniques, from classical to modern DL-based solutions, and testing the utility of the anonymized data.
We have tested the library on two main datasets:
1) Amazon dataset,
2) Enron emails dataset

We show it is able to generate anonymized corpora in both cases.

## Package High-Level Algorithm:

1) Data Preprocessing: tokenization, lemmatization, and stop word removal.
2) K-Anonymization: Generalization and Reduction.
3) Evaluation: Utilization is evaluated by embedding distance and semantic scores.
4) Visualization: dataset properties prior to and post anonymization will be visualized (optional)




# Getting Started
You can get started with kanonym4text immediately by installing it with pip:

## download package

```
pip install kanonym4text
```

## Step 1 - creat a data frame from your corpus
The code receives a data frame containing the corpus in the following format:
"txt" - column with the texts (default column name)

## Step 2 - import the following and initializethe object
Use [gensim word embedding model]([url](https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models)) for you choice (default - *glove-twitter-25*) 
```
from kanonym4text import Kanonym
kan = Kanonym()
```

## Step 3 - read the df:
```
import pandas as pd
df = pd.read_csv('YOUR_FILE.csv')
```

## Step 4 -  run the anonymization process on your corpus
```
dfa, dist = kan.anonymize(df, k=2)
```

**Running instructions:**
The main function is called *anonymize*. 

It's input parameters are:
---------------------------

* `df` - Input Dataframe
* `k` - k
* `col` - The column in df that holds the text to anonymize. Default - txt
* `num_stop` - Number of stop word to use. Default - 1000
* `num_jobs` - Number of CPUs to utilize. Default - 1. All CPUs - -1.
* `verbose` - Output text level. Default - 0.

The function outputs are:    
---------------------------
A tuple with these items:
1. A dataframe - the same df the user sent with additional columns:
> 1. general_txt - text after "generalization"
> 2. anon_txt_history - changes performed on text during annonymization process:    
>       [] - replaced     
>       {} - Lemmatize     
>       () - protected word (stop-word)     
> 3. anon_txt - resulted anonymized text
> 4. neighbors - indeces of k neigbors (Bow anonymized)

2. An averaged cosine distance of sentence embedding for the documents before and after the anonymizaion

# Running Example

use the following link to run some examples of the package on your own dataframe:

https://colab.research.google.com/drive/1eMSSvBxtsNFMOvKrUXgbsx1g3KOQD56s#scrollTo=ci2qjboGCt0A&uniqifier=1

or use the following code:
```
from kanonym4text import Kanonym
import pandas as pd

df = pd.read_csv('YOUR_FILE.csv')

kan = Kanonym('glove-twitter-25') # creating an object from class
dfa, dist = kan.anonymize(df, k=k, n_jobs=-1, plot=True)
```

# Support

## Create a Bug Report
If you see an error message or run into an issue, please create a bug report. This effort is valued and helps all users.

## Submit a Feature Request
If you have an idea, or you're missing a capability that would make development easier and more robust, please Submit a feature request.

If a similar feature request already exists, don't forget to leave a "+1". If you add some more information such as your thoughts and vision about the feature, your comments will be embraced warmly :)

# Contributing

Kanon4txt is an open-source project. We are committed to a fully transparent development process and highly appreciate any contributions. Whether you are helping us fix bugs, proposing new features, improving our documentation, or spreading the word - we would love to have you!




