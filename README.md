Company Mentor:  *Yair Horesh* 

Y-Data Mentor:  *Tom Haramaty*

Y-Data Students: *Hadas Neuman &  Lior Trieman*
## Overview:
In this project, we aim to apply data science techniques to anonymize textual data while preserving their utility. K-Anonymity is a technique used to ensure that an individual in a dataset cannot be identified by linking their attributes to external information, by forcing each row to be identical to k-1 other rows
The anonymized data can be used for various purposes like data sharing, research, and analysis without compromising privacy.
We plan on creating a novel algorithm for k-anonymity. Specifically, we address the case of unstructured data items, such as texts. Using various NLP techniques, from classical to modern DL based solutions, and test the utility of the anonymized data.
## Project Goals:
### Basic goal:
**Proof Of Concept** - Demonstrate it is possible to apply k-anonymity for very short texts while preserving functionality.

We plan to work on the “Amazon reviews” Dataset Amazon Reviews for Sentiment Analysis | Kaggle, and apply and model/pipeline that will ensure k-anonymization by:
●	Bag of words
●	Full identity (exactly the same text)

while trying to keep as much functional information we can.

we plan to test the utilization of this data set using two metrics, evaluated both before and after performing the anonymization:
●	A change in the sentiment score
●	% of preservation of keyphrase based on keyphrases extraction




### Advanced Goals:

●	Since this is relatively new task, we would like to gather some new insight on this task, and test how our methods generalize to a different dataset
●	Perform anonymization and utility tests on another data-set (Company dataset)
utility tests might be updated according to the data uses.
## Methodology:
1.	Data Preprocessing: data will be preprocessed using techniques like tokenization, stemming, and stop word removal or similar.
2.	K-Anonymization: we will develop a k-anonymity algorithm to be applied to the pre-processed textual data.
3.	Evaluation: The effectiveness of the k-anonymization technique will be evaluated using metrics like information loss, data utility and similar.
4.	Visualization: The anonymized dataset will be visualized to understand the degree of anonymity achieved and to check the utility of the dataset.
## Tools and Technologies:
1.	Programming Language: Python
2.	Libraries: Pandas, Numpy, Scikit-learn, Matplotlib, Seaborn
3.	Data: Amazon review dataset
4.	Algorithms: K-Anonymity Algorithm
5.	DevOps: Jupyter,Co-lab,Git/Github

## Major Milestones:

1.	POC of K-anonymization of Amazon dataset according to the matrics described above.
2.	Generalization of the algorithm to another dataset according to the company requirements.
## Expected Outcomes:
1.	An algorithm that enables anonymization of any given textual dataset.
2.	Anonymized textual dataset with preserved utility.
3.	Metrics for evaluating the effectiveness of the k-anonymity algorithm.
4.	Visualization of the anonymized dataset to check the degree of anonymity and the utility of the dataset.
 
