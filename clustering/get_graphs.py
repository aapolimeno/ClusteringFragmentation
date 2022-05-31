# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:17:22 2022

@author: Alessandra
"""

# pred labels = 0-8, so 9 should be added for rest 

import pandas as pd
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


df = pd.read_csv("../../data/new_split/new_test.csv", index_col = 0)
texts = df["text"].tolist()

tf_idfs = pd.read_csv("../../data/hlgd_predictions/baseline_test/tf_idf_reps_test.csv", index_col = 0)
cosines = pd.read_csv("../../data/hlgd_predictions/baseline_test/cosine_scores_test.csv", index_col = 0)
partitions = pd.read_csv("../../data/hlgd_predictions/baseline_test/partitions_test.csv", index_col = 0)


partitions = partitions.transpose()
partitions = partitions.rename(columns={0:'baseline_prediction'})


# Obtain the missed indeces which should be assigned to chain 9
pred_int = list(map(int, partitions.index.values))
gold_int = list(map(int, df.index.values))

pred_index = set(pred_int)
gold_index = set(gold_int)

different_indeces = pred_index.symmetric_difference(gold_index)

chain_9 = [931, 1092, 1078, 57, 31, 78, 335]

chain_9_df = pd.DataFrame(index = chain_9)
chain_9_df["baseline_prediction"] = 9

partitions = partitions.append(chain_9_df)

# partitions.to_csv("../../data/hlgd_predictions/baseline_test/full_pred_test.csv")

# Evaluation 

pred = partitions["baseline_prediction"].tolist()
gold = df["gold_label"].tolist()

hcv = homogeneity_completeness_v_measure(gold, pred)

