#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:13:11 2022

@author: alessandrapolimeno
"""

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix

def remove_punc(text):
    """
    This function removes punctuation from a string
    
    :param text: string 
    :return: cleaned string
    
    """
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in text:  
        if ele in punc:  
            text = text.replace(ele, " ") 
    return text


# ============= Load and preprocess data ============= 
# HLGD
hlgd_texts_train = pd.read_csv('../../data/hlgd_text_train.csv', index_col=0)
hlgd_links = hlgd_texts_train['url'].tolist()
hlgd_texts = hlgd_texts_train['text'].tolist()

# preprocessing 
hlgd_texts = [text.lower() for text in hlgd_texts] # convert to lowercase
hlgd_texts = [remove_punc(text) for text in hlgd_texts] # remove punctuation



# ============= Bag of Words ============= 

CountVec = CountVectorizer(ngram_range=(1,1), stop_words='english')
Count_data = CountVec.fit_transform(hlgd_texts)

hlgd_vectors = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())



# ============= Clustering ============= 

dendrogram = sc.dendrogram(sc.linkage(hlgd_vectors, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Articles')
plt.ylim(0, 1000)
plt.ylabel('Euclidean distances')
plt.show()


## Import clustering module

hca = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'ward')
y_hca = hca.fit_predict(hlgd_vectors)

## Visuvalization
# =============================================================================
# plt.scatter(hlgd_vectors[y_hca == 0, 0], hlgd_vectors[y_hca == 0, 1], s = 100, c = 'pink', label = 'Cluster 1')
# plt.scatter(hlgd_vectors[y_hca == 1, 0], hlgd_vectors[y_hca == 1, 1], s = 100, c = 'gray', label = 'Cluster 2')
# plt.scatter(hlgd_vectors[y_hca == 2, 0], hlgd_vectors[y_hca == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
# plt.scatter(hlgd_vectors[y_hca == 3, 0], hlgd_vectors[y_hca == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
# plt.scatter(hlgd_vectors[y_hca == 4, 0], hlgd_vectors[y_hca == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# plt.scatter(hlgd_vectors[y_hca == 5, 0], hlgd_vectors[y_hca == 5, 1], s = 100, c = 'blue', label = 'Cluster 6')
# plt.title('Clusters of articles')
# plt.xlabel("")
# plt.ylabel("")
# plt.legend()
# =============================================================================




# ============= Evaluation =============

# === Preprocessing ===

# Collect the predicted labels by matching them with their url
pred_data = pd.DataFrame(hlgd_links, columns=['url'])
pred_data['chain_id'] = y_hca

# Get labels and urls as lists 
pred_labels = pred_data['chain_id'].tolist()
pred_links = pred_data['url'].tolist()

# Load full data (needed for gold labels)
hlgd_texts_full = pd.read_csv('../../data/hlgd_train_full.csv', index_col=0)

# Extract all urls and gold labels as a dictionary 
# key: url, value: gold label
gold_url_a = dict(zip(hlgd_texts_full.url_a, hlgd_texts_full.timeline_id))
gold_url_b = dict(zip(hlgd_texts_full.url_b, hlgd_texts_full.timeline_id))

# Filter duplicats
gold_urls = []
gold_labels = []

for k, v in gold_url_a.items(): 
    gold_urls.append(k)
    gold_labels.append(v)


for k, v in gold_url_b.items(): 
    if k not in gold_urls:
        gold_urls.append(k)
        gold_labels.append(v)

# Add unique urls and labels to new dataframe
gold_chains = pd.DataFrame()
gold_chains['url'] = gold_urls
gold_chains['gold_label'] = gold_labels

# Merge gold and predicted labels by matching the urls 
eval_data = pred_data.merge(gold_chains, left_on='url', right_on='url')

# Write evaluation data out 
# eval_data.to_csv('../../data/eval_data_hlgd_t.csv', index = True)

# Create a dataframe for each predicted label to investigate the overlap 
df4 = eval_data.loc[eval_data['chain_id'] == 4]
df1 = eval_data.loc[eval_data['chain_id'] == 1]
df0 = eval_data.loc[eval_data['chain_id'] == 0]
df2 = eval_data.loc[eval_data['chain_id'] == 2]
df5 = eval_data.loc[eval_data['chain_id'] == 5]
df3 = eval_data.loc[eval_data['chain_id'] == 3]

# === Confusion matrix === 
true = eval_data['gold_label'].tolist()
pred = eval_data['chain_id'].tolist()


confusion_matrix(true, pred)
