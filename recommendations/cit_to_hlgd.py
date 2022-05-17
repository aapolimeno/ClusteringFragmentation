# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:22:47 2022

@author: Alessandra
"""
import pandas as pd


# ========== Load CI&T DeskDrop recommendations ========== 
# articles  
articles = pd.read_csv("../../data/CIT_DeskDrop/shared_articles.csv")
articles = articles[articles.lang != "pt"]
articles = articles[articles.eventType != "CONTENT REMOVED"]
del_articles = articles[1100:] 
articles = articles[:1100] # match the articles with number of HLGD articles

keep_articles = articles["contentId"].tolist()
deleted_arts = del_articles["contentId"].tolist()

# user interactions 
users = pd.read_csv("../../data/CIT_DeskDrop/users_interactions.csv")
unique_users = set(users['personId'].tolist())
unique_articles = set(articles['text'].tolist())


users = users[users["contentId"].isin(keep_articles) == True]
user_arts = users["contentId"].tolist()

#users.to_csv("../../data/users_interactions_filtered.csv")

# Load HLGD 
HLGD_texts = pd.read_csv("../../data/hlgd_texts.csv", index_col=0)
texts = HLGD_texts["text"].tolist()
gold_labels = HLGD_texts["gold_label"].tolist()
urls = HLGD_texts["url"].tolist()
articles["text"] = texts
articles["url"] = urls


content_ids = articles["contentId"].tolist()


cit_to_hlgd = pd.DataFrame()
cit_to_hlgd["content_id"] = content_ids
cit_to_hlgd["url"] = urls
cit_to_hlgd["text"] = texts
cit_to_hlgd["gold_label"] = gold_labels


#articles.to_csv("../../data/CIT_merged.csv")

cit_to_hlgd.to_csv("../../data/cit_to_hlgd.csv")

