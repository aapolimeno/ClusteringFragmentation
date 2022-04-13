# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 17:24:36 2022

@author: Alessandra
"""

import pandas as pd 
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
import numpy as np
from sklearn import metrics

# ========== Load evaluation data ==========
eval_data = pd.read_csv('../../data/eval_hlgd_tr_BERT.csv', index=True)


true = eval_data['gold_label'].tolist()
pred = eval_data['pred_label'].tolist()

# ========== V-Measure ==========

hcv = homogeneity_completeness_v_measure(true, pred)


# ============ Purity ============
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(true, pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)





# === Confusion matrix === 
confusion_matrix = confusion_matrix(true, pred)
df_cm = pd.DataFrame(confusion_matrix)
print(df_cm.to_latex(index_names = True ))

# === Precision, recall and F-score === 
prf = precision_recall_fscore_support(true, pred, average='micro')
print(prf)


# Full classification report 
target_names = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
report = classification_report(true, pred, target_names=target_names, output_dict=True, digits = 3)
df_report = pd.DataFrame(report, index = None).transpose()
print(df_report.to_latex(index=True, float_format="{:0.3f}".format))
