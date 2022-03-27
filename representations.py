#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:13:11 2022

@author: alessandrapolimeno
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


### Load data 
# HLGD
hlgd_texts = pd.read_csv('../data/hlgd_texts_train.csv', index_col=0)
hlgd_texts = hlgd_texts['text'].tolist()
hlgd_texts = [text.lower() for text in texts]

# 

CountVec = CountVectorizer(ngram_range=(1,1), stop_words='english')
Count_data = CountVec.fit_transform(texts)
hlgd_vectors = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())




