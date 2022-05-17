# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:22:46 2022

@author: Alessandra
"""

import pandas as pd

cit_to_hlgd = pd.read_csv("../../data/recommendations/cit_to_hlgd.csv", index_col = 0)
recs = pd.read_csv("../../data/recommendations/recs.csv", index_col = 0)

random_recs_df = pd.DataFrame(recs["random_recs"].tolist(), columns=["random_recs"])


# to do: 
# lists of story ids corresponding to the recommended articles (so order matters)

def convert_str_to_list(recs):
    """
    input: list of strings
    output: list of lists 
    
    """

    new_recs = []
    for rec in recs:
        rec = rec.strip('][').split(', ')
        rec = [int(article) for article in rec]
        new_recs.append(rec)
    return new_recs 


random_recs = recs["random_recs"].tolist()
# clean strings and add to list
random_recs = convert_str_to_list(random_recs)    



gold_labels = cit_to_hlgd["gold_label"].tolist()
content_ids = cit_to_hlgd["content_id"].tolist()

lookup_dict = {content_ids[i]: gold_labels[i] for i in range(len(content_ids))}


all_recs = []
for user in random_recs: 
    for rec in user:
        all_recs.append(rec)

set_recs = (set(content_ids).intersection(set(all_recs)))




for user in random_recs: 
    chain_recs = []
    for article in user: 
        print(article)
        chain = lookup_dict[article]
        print(chain)
        chain_recs.append(chain)
    print(chain_recs)


