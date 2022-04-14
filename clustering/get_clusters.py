# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:58:09 2022

@author: Alessandra
"""

import pandas as pd 
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
import spacy

### In case you have to download the SpaCy word embeddings, run this: 
    
#from spacy.cli import download
# download("en_core_web_md") 



# =================== Load data ===================
data = pd.read_csv('../../data/hlgd_texts.csv', index_col=0)
urls = data['url'].tolist()
sentences = data['text'].tolist()


# =================== Get embeddings ================================ 

# options: SBERT sentence embeds, SpaCy word embeds, BoW 


def get_BoW(sentences): 
    CountVec = CountVectorizer(ngram_range=(1,1), stop_words='english')
    Count_data = CountVec.fit_transform(sentences)
    bow_vectors = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())
    
    return bow_vectors


def get_representation(sentences, method = "word"):     
    if method == "SBERT":
        model = SentenceTransformer('all-MiniLM-L6-v2') 
        embeddings = model.encode(sentences)
        
    if method == "word": 
        embeddings = []
        nlp = spacy.load('en_core_web_md')
        for sent in sentences: 
           doc = nlp(sent)
           embeddings.append(doc.vector) 
    
    if method == "BoW": 
        embeddings = get_BoW(sentences)
    
    return embeddings
    

def get_clusters(embeddings, method, dev = 0):
    
    if method == "BoW": 
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold = 200 ,linkage = 'ward')
    else: 
        if dev == 0: 
            n_clusters = 8
        else: 
            n_clusters = 2
        clustering_model = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward') #, affinity='cosine', linkage='average', distance_threshold=0.4)
    
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_
    
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
    
        clustered_sentences[cluster_id].append(sentences[sentence_id])
        
    return cluster_assignment



# ============ Represent the articles with the desired method ============
# Options: SBERT (sentence embeddings), word (word embeddings), BoW (Bag of Words)
methods = ["SBERT", "word", "BoW"]

pred_clusters = pd.DataFrame(urls, columns = ['url'])

for method in methods:
    print("===================================================================")
    print(f"Get article representation with method {method}...")
    embeddings = get_representation(sentences, method = method)
    print(f"Performing agglomerative hierarchical clustering for {method} representations...")
    clusters = get_clusters(embeddings, method, dev = 0)
    print(f"Save {method} clustering outcome...")
    pred_clusters[f'{method}_pred'] = clusters
print()
print("===================================================================")
print("All done!")
print("===================================================================")
    
print(set(clusters))

# ============ Save ============ 
pred_clusters.to_csv('../../data/hlgd_predictions/predictions_raw.csv', index = True)


