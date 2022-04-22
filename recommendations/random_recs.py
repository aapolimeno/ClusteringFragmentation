#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:36:39 2022

@author: alessandrapolimeno
"""

import pandas as pd 
import random


# ============================= Load data =============================
full_data = pd.read_csv("../../data/eval_data_hlgd_t.csv")
full_data = full_data.drop(["Unnamed: 0", "chain_id"],
                           axis=1) # drop irrelevant columns

index = full_data.index
full_data["article_id"] = index # add column with article ids based on index
full_data.to_csv("../../data/article_ids_train.csv")

# ================== Generate random recommendations ================== 

users = [i for i in range(717)] # 100 users 

random_recs = pd.DataFrame()
random_recs["user_id"] = users

recs = []

for user in users: 
    rec = random.choices(index, k = 20) # 15 recs per user 
    recs.append(rec)

random_recs["recs"] = recs
random_recs.to_csv("../../data/random_recs.csv", index=False)

