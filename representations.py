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
hlgd_texts = pd.read_csv('../../data/hlgd_texts_train.csv', index_col=0)
hlgd_texts = hlgd_texts['text'].tolist()

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
hca = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage = 'ward')
y_hca = hca.fit_predict(hlgd_vectors)

## Visuvalization
plt.scatter(hlgd_vectors[y_hca == 0, 0], hlgd_vectors[y_hca == 0, 1], s = 100, c = 'pink', label = 'Cluster 1')
plt.scatter(hlgd_vectors[y_hca == 1, 0], hlgd_vectors[y_hca == 1, 1], s = 100, c = 'gray', label = 'Cluster 2')
plt.scatter(hlgd_vectors[y_hca == 2, 0], hlgd_vectors[y_hca == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(hlgd_vectors[y_hca == 3, 0], hlgd_vectors[y_hca == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(hlgd_vectors[y_hca == 4, 0], hlgd_vectors[y_hca == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(hlgd_vectors[y_hca == 5, 0], hlgd_vectors[y_hca == 5, 1], s = 100, c = 'blue', label = 'Cluster 6')
plt.scatter(hlgd_vectors[y_hca == 6, 0], hlgd_vectors[y_hca == 6, 1], s = 100, c = 'salmon', label = 'Cluster 7')
plt.title('Clusters of articles')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

