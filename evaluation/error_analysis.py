# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:03:23 2022

@author: Alessandra
"""

import pandas as pd

data = pd.read_csv("../../data/hlgd_predictions/best/preds_test_final.csv", index_col = 0)

gold = set(data["gold"].tolist())
sbert_ac = set(data["SBERT_AC"].tolist())
sbert_db = set(data["SBERT_DBScan"].tolist())
word_ac = set(data["word_AC"].tolist())

# GOLD 
c2 =  data.loc[data['gold'] == 2]
c4 =  data.loc[data['gold'] == 4]
c5 =  data.loc[data['gold'] == 5]
c6 =  data.loc[data['gold'] == 6]
c7 =  data.loc[data['gold'] == 7]
c8 =  data.loc[data['gold'] == 8]
c9 =  data.loc[data['gold'] == 9]

sbert_ac = c9["SBERT_AC"].tolist()
word_ac = c9["word_AC"].tolist()

# sbert: 4
# word: 1
wrong = []
for pred in sbert_ac: 
    if pred != 3: 
        wrong.append(pred)


mistakes = []
for tup in zip(sbert_ac, word_ac): 
    if tup[0] != 3 and tup[1] != 6: 
        mistakes.append(tup)
        

# SBERT_AHC 
s0 = data.loc[data["SBERT_AC"] == 0]
s1 = data.loc[data["SBERT_AC"] == 1]
s2 = data.loc[data["SBERT_AC"] == 2]
s3 = data.loc[data["SBERT_AC"] == 3]
s4 = data.loc[data["SBERT_AC"] == 4]
s5 = data.loc[data["SBERT_AC"] == 5]
s6 = data.loc[data["SBERT_AC"] == 6]
s7 = data.loc[data["SBERT_AC"] == 7]
s8 = data.loc[data["SBERT_AC"] == 8]

# WORD_AC 
word0 = data.loc[data["word_AC"] == 0]
word1 = data.loc[data["word_AC"] == 1]
word2 = data.loc[data["word_AC"] == 2]
word3 = data.loc[data["word_AC"] == 3]
word4 = data.loc[data["word_AC"] == 4]
word5 = data.loc[data["word_AC"] == 5]
word6 = data.loc[data["word_AC"] == 6]
word7 = data.loc[data["word_AC"] == 7]
word8 = data.loc[data["word_AC"] == 8]


# SBERT_DB 
db0 = data.loc[data["SBERT_DBScan"] == 0]
db1 = data.loc[data["SBERT_DBScan"] == 1]
db2 = data.loc[data["SBERT_DBScan"] == 2]
db3 = data.loc[data["SBERT_DBScan"] == 3]
db4 = data.loc[data["SBERT_DBScan"] == -1]