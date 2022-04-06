# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:15:22 2022

@author: Alessandra
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix


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

# =============================================================================
# for i, cluster in clustered_sentences.items():
#     print("Cluster ", i+1)
#     print(cluster)
#     print("")
#     
# =============================================================================
    
pred_clusters = pd.DataFrame(urls, columns = ['url'])
pred_clusters['pred_label'] = cluster_assignment


# ============= Evaluate ============= 

eval_data = pd.read_csv("../../data/eval_data_hlgd_t.csv")

eval_data = eval_data.merge(pred_clusters, left_on='url', right_on='url')
eval_data = eval_data.drop(['chain_id'], axis = 1)
eval_data = eval_data.drop(['Unnamed: 0'], axis = 1)

# Create a dataframe for each predicted label to investigate the overlap 
df4 = eval_data.loc[eval_data['pred_label'] == 4]
df1 = eval_data.loc[eval_data['pred_label'] == 1]
df0 = eval_data.loc[eval_data['pred_label'] == 0]
df2 = eval_data.loc[eval_data['pred_label'] == 2]
df5 = eval_data.loc[eval_data['pred_label'] == 5]
df3 = eval_data.loc[eval_data['pred_label'] == 3]

# Convert gold labels to predicted format 
convert_dict = {1:0, 4:1, 2:2, 9:3, 3:4, 6:5}
gold_labels = eval_data['gold_label'].tolist()

converted_labels = []
for label in gold_labels: 
    label = convert_dict[label]
    converted_labels.append(label)
    
eval_data['gold_label'] = converted_labels

# Save model output
# eval_data.to_csv('../../data/eval_hlgd_tr_BERT.csv', index = True)

# === Confusion matrix === 
true = eval_data['gold_label'].tolist()
pred = eval_data['pred_label'].tolist()

confusion_matrix = confusion_matrix(true, pred)
df_cm = pd.DataFrame(confusion_matrix)
print(df_cm.to_latex(index_names = True ))

# === Precision, recall and F-score === 
# precision_recall_fscore_support(true, pred, average=None, labels = [0,1,2,3,4,5])
target_names = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
report = classification_report(true, pred, target_names=target_names, output_dict=True, digits = 3)
df_report = pd.DataFrame(report, index = None).transpose()
print(df_report.to_latex(index=True, float_format="{:0.3f}".format))
