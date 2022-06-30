# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:04:40 2022

@author: Alessandra
"""

import pandas as pd

from matplotlib import pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize

data = pd.read_csv("../../data/new_split/new_test.csv", index_col = 0)
dev_data = pd.read_csv("../../data/new_split/new_dev.csv", index_col = 0)

c0 =  dev_data.loc[dev_data['gold_label'] == 0]
c1 =  dev_data.loc[dev_data['gold_label'] == 1]
c2 =  data.loc[data['gold_label'] == 2]
c3 =  dev_data.loc[dev_data['gold_label'] == 3]
c4 =  data.loc[data['gold_label'] == 4]
c5 =  data.loc[data['gold_label'] == 5]
c6 =  data.loc[data['gold_label'] == 6]
c7 =  data.loc[data['gold_label'] == 7]
c8 =  data.loc[data['gold_label'] == 8]
c9 =  data.loc[data['gold_label'] == 9]



def get_mean_tokens(chain_df):

    texts = chain_df['text']
    toks = []
    for text in texts: 
        tokens = word_tokenize(text)
        for tok in tokens: 
            toks.append(tok)
           
    mean = len(toks) / len(texts)
    
    return mean


def get_mean_length(chain_df): 
    texts = chain_df["text"]
    lens = []
    for text in texts: 
        length = len(text)
        lens.append(length)
        
    mean = np.mean(lens)
        
    return mean



all_chains = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]


all_means = []
for chain in all_chains: 
    mean = get_mean_tokens(chain)
    all_means.append(mean)
    

all_mean_lengths = []
for chain in all_chains: 
    mean_len = get_mean_length(chain)
    all_mean_lengths.append(mean_len)

# pie chart 
len_chains = [99, 195, 167, 69, 163, 152, 83, 99, 126, 241]
labels = [0,1,2,3,4,5,6,7,8,9]

fig = plt.figure(figsize =(10, 7))
theme = plt.get_cmap('Accent_r')
plt.legend(len_chains, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()
plt.pie(len_chains)

