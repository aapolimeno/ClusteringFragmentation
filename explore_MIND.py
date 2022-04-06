# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 11:11:15 2022

@author: alessandrapolimeno
"""

import pandas as pd

# ========== Load MIND recommendations ========== 
path = "../../data/MINDsmall_dev/behaviors.tsv"
behaviors = pd.read_csv(path, delimiter = "\t", header = None, index_col = 0)
behaviors.columns = ["user_id", "date_time", "recs", "clicks"]
behaviors = behaviors.dropna()

user_ids = behaviors["user_id"].tolist()
uis_set = set(user_ids)


