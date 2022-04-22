# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:18:35 2022

@author: Alessandra
"""

import pandas as pd
import random

hybrid_recs = pd.read_csv("../../data/recommendations/recs_hybrid.csv")
popular_recs = pd.read_csv("../../data/recommendations/recs_pop.csv")

recs = hybrid_recs.merge(popular_recs, left_on = "person_id", right_on = "person_id")

pop_recs = recs["popular_recs"].tolist()
hybrid_recs = recs["hybrid_recs"].tolist()

# ================ Convert recommendations to list ================

def convert_str_to_list(recs):
    new_recs = []
    for rec in recs:
        rec = rec.strip('][').split(', ')
        rec = [int(article) for article in rec]
        new_recs.append(rec)
    return new_recs 

pop_recs = convert_str_to_list(pop_recs)
hybrid_recs = convert_str_to_list(hybrid_recs)        
    
# ================ Remove recommendations at rank lower than 20 ================

def filter_recs(recs):
    filtered_recs = []
    for rec in recs: 
        rec = rec[:20]
        filtered_recs.append(rec)
    return filtered_recs

filtered_pop_recs = filter_recs(pop_recs)
filtered_hybrid_recs = filter_recs(hybrid_recs)

recs["popular_recs"] = filtered_pop_recs
recs["hybrid_recs"] = filtered_hybrid_recs


# ================ Generate random recommendations ================

def get_unique_articles(recs): 
    unique_articles = set()
    for rec in recs: 
        for article in rec: 
            unique_articles.add(article)
    return list(unique_articles)


# Get unique articles 
all_articles = get_unique_articles(filtered_pop_recs)
all_hybrid_articles = get_unique_articles(filtered_hybrid_recs)
all_articles.extend(all_hybrid_articles)

# Generate recs for each user 
users = recs["person_id"].tolist()

random_recs = []
for user in users: 
    rec = random.choices(all_articles, k = 20) # 15 recs per user 
    random_recs.append(rec)
    
recs["random_recs"] = random_recs


recs.to_csv("../../data/recommendations/recs.csv", index=False)
