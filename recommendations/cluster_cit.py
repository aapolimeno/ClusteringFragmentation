# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:30:24 2022

@author: Alessandra
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load data
cit_clusters = pd.read_csv("../../data/CIT_DeskDrop/clustered/CIT_clusters.csv", index_col=0)
cit_clusters = cit_clusters.transpose()
cit_clusters = cit_clusters.rename(columns = {0:"cluster"})

hlgd_clusters = pd.read_csv("../../data/CIT_DeskDrop/clustered/hlgd_clusters.csv", index_col=0)
hlgd_clusters = hlgd_clusters.transpose()
hlgd_clusters = hlgd_clusters.rename(columns = {0:"cluster"})

# Inspect clusters 
cit_set = set(cit_clusters["cluster"].tolist())
cit_0 = cit_clusters.loc[cit_clusters["cluster"] == 0]
cit_1 = cit_clusters.loc[cit_clusters["cluster"] == 1]
cit_2 = cit_clusters.loc[cit_clusters["cluster"] == 2]
cit_3 = cit_clusters.loc[cit_clusters["cluster"] == 3]
cit_4 = cit_clusters.loc[cit_clusters["cluster"] == 4]
cit_5 = cit_clusters.loc[cit_clusters["cluster"] == 5]


hlgd_set = set(hlgd_clusters["cluster"].tolist())
hlgd_0 = hlgd_clusters.loc[hlgd_clusters["cluster"] == 0]
hlgd_1 = hlgd_clusters.loc[hlgd_clusters["cluster"] == 1]
hlgd_2 = hlgd_clusters.loc[hlgd_clusters["cluster"] == 2]
hlgd_3 = hlgd_clusters.loc[hlgd_clusters["cluster"] == 3]
hlgd_4 = hlgd_clusters.loc[hlgd_clusters["cluster"] == 4]
hlgd_5 = hlgd_clusters.loc[hlgd_clusters["cluster"] == 5]
hlgd_6 = hlgd_clusters.loc[hlgd_clusters["cluster"] == 6]
hlgd_7 = hlgd_clusters.loc[hlgd_clusters["cluster"] == 7]
hlgd_8 = hlgd_clusters.loc[hlgd_clusters["cluster"] == 8]
hlgd_9 = hlgd_clusters.loc[hlgd_clusters["cluster"] == 9]

