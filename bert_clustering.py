# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:15:22 2022

@author: Alessandra
"""

import numpy as np
import pandas as pd 
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


# ============= Load data =============
hlgd_texts_train = pd.read_csv('../../data/hlgd_text_train.csv', index_col=0)
urls = hlgd_texts_train['url'].tolist()
sentences = hlgd_texts_train['text'].tolist()


# ============= Get sentence embeddings ============= 

model = SentenceTransformer('all-MiniLM-L6-v2') 
sentence_embeddings = model.encode(sentences)


# ============= Apply clustering ============= 

clustering_model = AgglomerativeClustering(n_clusters=6, linkage = 'ward') #, affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(sentence_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(sentences[sentence_id])

for i, cluster in clustered_sentences.items():
    print("Cluster ", i+1)
    print(cluster)
    print("")
    
    
pred_clusters = pd.DataFrame(urls, columns = ['url'])
pred_clusters['pred_chain'] = cluster_assignment


# ============= Evaluate ============= 

eval_data = pd.read_csv("../../data/eval_data_hlgd_t.csv")

eval_data = eval_data.merge(pred_clusters, left_on='url', right_on='url')
eval_data = eval_data.drop(['chain_id'], axis = 1)
eval_data = eval_data.drop(['Unnamed: 0'], axis = 1)

# Create a dataframe for each predicted label to investigate the overlap 
df4 = eval_data.loc[eval_data['pred_chain'] == 4]
df1 = eval_data.loc[eval_data['pred_chain'] == 1]
df0 = eval_data.loc[eval_data['pred_chain'] == 0]
df2 = eval_data.loc[eval_data['pred_chain'] == 2]
df5 = eval_data.loc[eval_data['pred_chain'] == 5]
df3 = eval_data.loc[eval_data['pred_chain'] == 3]

