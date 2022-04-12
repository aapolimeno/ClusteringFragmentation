# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:15:22 2022

@author: Alessandra
"""
import pandas as pd 
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
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



# ============= Load data =============
hlgd_texts_train = pd.read_csv('../../data/hlgd_text_train.csv', index_col=0)
urls = hlgd_texts_train['url'].tolist()
sentences = hlgd_texts_train['text'].tolist()

# Preprocessing 
sentences = [text.lower() for text in sentences] # convert to lowercase
sentences = [remove_punc(text) for text in sentences] # remove punctuation

# ============= Get BoW representations ============= 

CountVec = CountVectorizer(ngram_range=(1,1), stop_words='english')
Count_data = CountVec.fit_transform(sentences)

bow_vectors = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())


# ============= Apply clustering ============= 

clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold = 197 ,linkage = 'ward') #, affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(bow_vectors)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(sentences[sentence_id])

print(set(cluster_assignment))

# =============================================================================
# for i, cluster in clustered_sentences.items():
#     print("Cluster ", i+1)
#     print(cluster)
#     print("")
#     
# =============================================================================
    
pred_clusters = pd.DataFrame(urls, columns = ['url'])
pred_clusters['pred_label'] = cluster_assignment


# ============= Prepare evaluation ============= 

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





from sklearn.metrics import classification_report, precision_recall_fscore_support


# === Confusion matrix === 
true = eval_data['gold_label'].tolist()
pred = eval_data['pred_label'].tolist()

confusion_matrix = confusion_matrix(true, pred)
df_cm = pd.DataFrame(confusion_matrix)
print(df_cm.to_latex(index_names = True ))

# === Precision, recall and F-score === 
prf = precision_recall_fscore_support(true, pred, average='micro')
print(prf)


# Full classification report 
target_names = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
report = classification_report(true, pred, target_names=target_names, output_dict=True, digits = 3)
df_report = pd.DataFrame(report, index = None).transpose()
print(df_report.to_latex(index=True, float_format="{:0.3f}".format))


