#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:13:11 2022

@author: alessandrapolimeno
"""

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import scipy.cluster.hierarchy as sc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def remove_punc(text):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in text:  
        if ele in punc:  
            text = text.replace(ele, " ") 
    return text



### Load data 
# HLGD
hlgd_texts_train = pd.read_csv('../../data/hlgd_text_train.csv', index_col=0)
hlgd_links = hlgd_texts_train['url'].tolist()
hlgd_texts = hlgd_texts_train['text'].tolist()

# preprocessing 
hlgd_texts = [text.lower() for text in hlgd_texts] # convert to lowercase
hlgd_texts = [remove_punc(text) for text in hlgd_texts] # remove punctuation

### Create BoW model
CountVec = CountVectorizer(ngram_range=(1,1), stop_words='english')
Count_data = CountVec.fit_transform(hlgd_texts)

hlgd_vectors = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())

### Clustering 

dendrogram = sc.dendrogram(sc.linkage(hlgd_vectors, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Articles')
plt.ylim(0, 1000)
plt.ylabel('Euclidean distances')
plt.show()


## Import clustering module
from sklearn.cluster import AgglomerativeClustering
hca = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'ward')
y_hca = hca.fit_predict(hlgd_vectors)

## Visuvalization
plt.scatter(hlgd_vectors[y_hca == 0, 0], hlgd_vectors[y_hca == 0, 1], s = 100, c = 'pink', label = 'Cluster 1')
plt.scatter(hlgd_vectors[y_hca == 1, 0], hlgd_vectors[y_hca == 1, 1], s = 100, c = 'gray', label = 'Cluster 2')
plt.scatter(hlgd_vectors[y_hca == 2, 0], hlgd_vectors[y_hca == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(hlgd_vectors[y_hca == 3, 0], hlgd_vectors[y_hca == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(hlgd_vectors[y_hca == 4, 0], hlgd_vectors[y_hca == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(hlgd_vectors[y_hca == 5, 0], hlgd_vectors[y_hca == 5, 1], s = 100, c = 'blue', label = 'Cluster 6')
plt.title('Clusters of articles')
plt.xlabel("")
plt.ylabel("")
plt.legend()


### Evaluation 

pred_data = pd.DataFrame(hlgd_links, columns=['url'])
pred_data['chain_id'] = y_hca

pred_labels = pred_data['chain_id'].tolist()
pred_links = pred_data['url'].tolist()

print(set(y_hca))
print(set(pred_labels))

# load full data 
hlgd_texts_full = pd.read_csv('../../data/hlgd_train_full.csv', index_col=0)

gold_url_a = dict(zip(hlgd_texts_full.url_a, hlgd_texts_full.timeline_id))
gold_url_b = dict(zip(hlgd_texts_full.url_b, hlgd_texts_full.timeline_id))

gold_urls = []
gold_labels = []

for k, v in gold_url_a.items(): 
    gold_urls.append(k)
    gold_labels.append(v)


for k, v in gold_url_b.items(): 
    if k not in gold_urls:
        gold_urls.append(k)
        gold_labels.append(v)

gold_chains = pd.DataFrame()
gold_chains['url'] = gold_urls
gold_chains['gold_label'] = gold_labels


eval_data = pred_data.merge(gold_chains, left_on='url', right_on='url')


eval_data.to_csv('../../data/eval_data_hlgd_t.csv', index = True)

true = eval_data['gold_label'].tolist()
pred = eval_data['chain_id'].tolist()

print("true: ", set(true))
print("pred: ", set(pred))



print(eval_data.loc[eval_data['chain_id'] == 3])
print(eval_data.loc[eval_data['chain_id'] == 5])


df4 = eval_data.loc[eval_data['chain_id'] == 4]
df1 = eval_data.loc[eval_data['chain_id'] == 1]
df0 = eval_data.loc[eval_data['chain_id'] == 0]
df2 = eval_data.loc[eval_data['chain_id'] == 2]

# confusion matrix 
from sklearn.metrics import confusion_matrix
confusion_matrix(true, pred)
