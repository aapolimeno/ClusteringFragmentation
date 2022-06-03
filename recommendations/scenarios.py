# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:28:21 2022

@author: Alessandra
"""

import pandas as pd 
import random 
# Load data 
# gold = pd.read_csv("../../data/new_split/new_test.csv", index_col = 0)
pred = pd.read_csv("../../data/hlgd_predictions/best/preds_test_final.csv", index_col = 0)


# Get list of each gold story chain 
chain2 = pred.loc[pred["gold"] == 2]
chain4 = pred.loc[pred["gold"] == 4]
chain5 = pred.loc[pred["gold"] == 5]
chain6 = pred.loc[pred["gold"] == 6]
chain7 = pred.loc[pred["gold"] == 7]
chain8 = pred.loc[pred["gold"] == 8]
chain9 = pred.loc[pred["gold"] == 9]

# ============= Scenario 1: Low Fragmentation =============
# ===== 1a: 8 recommendations per user =====
low_frag_8 = pd.DataFrame()
low_frag_8 = chain2.sample()
low_frag_8 = low_frag_8.append(chain4.sample())


users = [i for i in range(1000)]
low_frag_8["user"] = users
urls = pred["url"].tolist()
for i in range(1000): 
    sample = random.sample(urls, 20)
    
    
    

# ============= Scenario 3: High Fragmentation =============