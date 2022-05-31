# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:58:09 2022

@author: Alessandra
"""

import pandas as pd 
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from nltk import tokenize
import tensorflow_hub as hub

### In case you have to download the SpaCy word embeddings, run this: 
    
#from spacy.cli import download
# download("en_core_web_md") 

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# =================== Load data ===================
data = pd.read_csv('../../data/new_split/new_test.csv', index_col=0)
#data = pd.read_csv('../../data/new_split/new_dev.csv', index_col=0)
data = data[:2]
urls = data['url'].tolist()
sentences = data['text'].tolist()




# =================== Get embeddings =================== 

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
           
    if method == "word_sent":
        
        sents = []

        for sent in sentences: 
                sentence = tokenize.sent_tokenize(sent)
                sents.append(sentence)
                
        embeddings = embed(sentences)

        print("embeddings: ", embeddings)

        
    
    if method == "BoW": 
        embeddings = get_BoW(sentences)
    
    return embeddings
    

def get_clusters(embeddings, method, distance, alg = "AC"):
    
    
    if alg == "AC":

# =============================================================================
#         if method == "BoW" :
#             distance_threshold = 250
#         else: 
#             distance_threshold = 5
# =============================================================================
            
        clustering_model = AgglomerativeClustering(n_clusters = None, linkage = 'ward', distance_threshold = distance) #, affinity='cosine', linkage='average', distance_threshold=0.4)
    
        clustering_model.fit(embeddings)
        cluster_assignment = clustering_model.labels_
        
        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []
        
            clustered_sentences[cluster_id].append(sentences[sentence_id])
        
    if alg == "DBScan": 
        
        if method == "SBERT": 
            ep = 1
            min_samples = 96.5
        if method == "BoW": 
            ep = 0.5
            min_samples = 41.5
        if method == "word": 
            ep = 14.5
            min_samples = 0.5
        
        clustering = DBSCAN(eps=ep, min_samples=min_samples).fit(embeddings)
        cluster_assignment = clustering.labels_
    
    return cluster_assignment



# ============ Represent the articles with the desired method ============
# Options: SBERT (sentence embeddings), word (word embeddings), BoW (Bag of Words)
#methods = ["SBERT", "word", "BoW"]
methods = ["word_sent"]

pred_clusters = pd.DataFrame(urls, columns = ['url'])

#distance_thresholds = [4,5,6,7,8,9,124,134,144]


true = data["gold_label"].tolist()

labels = []
homs = []
comps = []
v_measures = []
sils = []
dbs = []
chs = []

distance = 2

for method in methods:
        print("===================================================================")
        print(f"Get article representation with method {method}...")
        embeddings = get_representation(sentences, method = method)
        print(f"Performing agglomerative hierarchical clustering for {method} representations...")
       
        clusters = get_clusters(embeddings, method, distance, alg = "AC")
        #print(set(clusters))
        print(f"Save {method} clustering outcome...")
        pred_clusters[f'{method}_AC_{distance}'] = clusters
        
        # Calculate V-measure 
        
        pred = clusters
        label = f"{method}_AC_{distance}"
        hcv = homogeneity_completeness_v_measure(true, pred)
        if len(set(clusters)) > 1: 
            sil = silhouette_score(embeddings, clusters, metric="euclidean")
            db = davies_bouldin_score(embeddings, clusters)
            ch = calinski_harabasz_score(embeddings, clusters)
        else: 
            sil = 0
            db = 0
            ch = 0
            
        df = pd.DataFrame(columns = ["model", "homogeneity", "completeness", "v_measure", "sil", "db", "ch"])
        
        add = [label, hcv[0], hcv[1], hcv[2], sil, db, ch]
        
        labels.append(label)
        homs.append(hcv[0])
        comps.append(hcv[1])
        v_measures.append(hcv[2])
        sils.append(sil)
        dbs.append(db)
        chs.append(ch)


eval_scores = pd.DataFrame()
eval_scores["model"] = labels 
eval_scores["hom"] = homs
eval_scores["comp"] = comps
eval_scores["v_measure"] = v_measures
eval_scores["sil"] = sils 
eval_scores["db"] = dbs 
eval_scores["ch"] = chs   

print()
print("===================================================================")
print("All done!")
print("===================================================================")



    
#pred_clusters.to_csv('../../data/hlgd_predictions/evaluation_AC_test/preds_test_3_7.csv', index = True)
#eval_scores.to_csv("../../data/hlgd_predictions/evaluation_AC_test/eval_scores_test_3_7.csv", index = True)

# best_sbert = pred_clusters["SBERT_AC_5"].tolist()
# best_word = pred_clusters["word_AC_4"].tolist()
# best_bow = pred_clusters["BoW_AC_124"].tolist()
# best_preds = pd.DataFrame()
# best_preds["SBERT_ACH"] = best_sbert
# best_preds["word_AHC"] = best_word
# best_preds["BoW_AHC"] = best_bow
# best_preds["gold"] = true


#best_preds.to_csv("../../data/hlgd_predictions/evaluation_AC_test/best_test_3_7.csv")


# best_eval_scores = pd.DataFrame()
#
#
# best_sbert_scores = eval_scores.loc[eval_scores["model"] == "SBERT_AC_5"]
# best_word_scores = eval_scores.loc[eval_scores["model"] == "word_AC_4"]
# best_bow_scores = eval_scores.loc[eval_scores["model"] == "BoW_AC_124"]
#
#
# best_eval_scores = best_eval_scores.append(best_sbert_scores, ignore_index = True)
# best_eval_scores = best_eval_scores.append(best_word_scores, ignore_index = True)
# best_eval_scores = best_eval_scores.append(best_bow_scores, ignore_index = True)

# best_eval_scores.to_csv("../../data/hlgd_predictions/evaluation_AC_test/best_eval_scores_test_3_7.csv")


#v_measure.to_csv("../../data/hlgd_predictions/eval_scores_devbig.csv")


#pred = pred_clusters["word_pred"]
#hcv = homogeneity_completeness_v_measure(true, pred)
    
# print(set(clusters))

# ============ Save ============ 
#pred_clusters.to_csv('../../data/hlgd_predictions/predictions_raw.csv', index = True)


