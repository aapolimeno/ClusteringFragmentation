# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:00:58 2022

@author: Alessandra
"""

import pandas as pd 

# ========== Load evaluation and gold data ==========

data = pd.read_csv('../../data/hlgd_texts.csv', index_col=0)

eval_data = pd.read_csv('../../data/hlgd_predictions/predictions_raw.csv', index_col = 0)
gold_data = pd.read_csv('../../data/hlgd_predictions/gold_labels.csv', index_col=0)

eval_data = eval_data.merge(gold_data, left_on='url', right_on='url')

pred_SBERT = eval_data['SBERT_pred'].tolist()
pred_word = eval_data['word_pred'].tolist()
pred_bow = eval_data['BoW_pred'].tolist()

# Inspect each cluster 

c2 =  eval_data.loc[eval_data['gold_label'] == 2]
c3 =  eval_data.loc[eval_data['gold_label'] == 3]
c4 =  eval_data.loc[eval_data['gold_label'] == 4]
c5 =  eval_data.loc[eval_data['gold_label'] == 5]
c6 =  eval_data.loc[eval_data['gold_label'] == 6]
c7 =  eval_data.loc[eval_data['gold_label'] == 7]
c8 =  eval_data.loc[eval_data['gold_label'] == 8]
c9 =  eval_data.loc[eval_data['gold_label'] == 9]


c0 =  eval_data.loc[eval_data["BoW_pred"] == 0]
c1 =  eval_data.loc[eval_data["BoW_pred"] == 1]
c2 =  eval_data.loc[eval_data['BoW_pred'] == 2]
c3 =  eval_data.loc[eval_data['BoW_pred'] == 3]
c4 =  eval_data.loc[eval_data['BoW_pred'] == 4]
c5 =  eval_data.loc[eval_data['BoW_pred'] == 5]
c6 =  eval_data.loc[eval_data['BoW_pred'] == 6]
c7 =  eval_data.loc[eval_data['BoW_pred'] == 7]


c0 =  eval_data.loc[eval_data["word_pred"] == 0]
c1 =  eval_data.loc[eval_data["word_pred"] == 1]
c2 =  eval_data.loc[eval_data['word_pred'] == 2]
c3 =  eval_data.loc[eval_data['word_pred'] == 3]
c4 =  eval_data.loc[eval_data['word_pred'] == 4]
c5 =  eval_data.loc[eval_data['word_pred'] == 5]
c6 =  eval_data.loc[eval_data['word_pred'] == 6]
c7 =  eval_data.loc[eval_data['word_pred'] == 7]


c0 =  eval_data.loc[eval_data["SBERT_pred"] == 0]
c1 =  eval_data.loc[eval_data["SBERT_pred"] == 1]
c2 =  eval_data.loc[eval_data['SBERT_pred'] == 2]
c3 =  eval_data.loc[eval_data['SBERT_pred'] == 3]
c4 =  eval_data.loc[eval_data['SBERT_pred'] == 4]
c5 =  eval_data.loc[eval_data['SBERT_pred'] == 5]
c6 =  eval_data.loc[eval_data['SBERT_pred'] == 6]
c7 =  eval_data.loc[eval_data['SBERT_pred'] == 7]


filter_indeces = c6.index.tolist()

filter_sents = data.iloc[[ind for ind in filter_indeces]]
for num, sent in enumerate(filter_sents['text'].tolist()):
    print(num)
    print(sent)
    print()


# ============ Convert labels to gold format ============
def get_conversion_dict(method, dev = 0):
    
    if dev == 0: 
    
        if method == "word": 
            convert_dict = {9:1, 1:0 , 2:2, 3:3, 6:4, 4:5}
        if method == "SBERT": 
            convert_dict = {1:0, 4:1, 2:2, 9:3, 3:4, 6:5}
            
    if dev == 1: 
        convert_dict = {0:1, 1:0}
        #convert_dict = {1:2, 0:1 2:0}
            
    return convert_dict

def perform_conversion(method): 
    
    convert_dict = get_conversion_dict(method, dev = 1)
    
    converted_labels = []
    
    if method == "word": 
        for label in pred_word: 
            label = convert_dict[label]
            converted_labels.append(label)
            
    if method == "SBERT": 
        for label in pred_SBERT: 
            label = convert_dict[label]
            converted_labels.append(label)
    
    if method == "BoW": 
        for label in pred_bow: 
            label = convert_dict[label]
            converted_labels.append(label)
    
    eval_data[f"{method}_pred"] = converted_labels

methods = ["SBERT", "word", "BoW"]

print("==================================================")

for method in methods: 
    print(f"converting the labels of {method} representations...")
    perform_conversion(method)

print("Done!")  
print("==================================================")  

eval_data.to_csv('../../data/hlgd_predictions/predictions_dev.csv', index = True)
