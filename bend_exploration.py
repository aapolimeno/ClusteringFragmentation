#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 13:19:18 2022

@author: alessandrapolimeno
"""

import pandas as pd
import numpy as np 
from numpy import mean
import nltk  
import random


# step 1: read in the data 

ben_raw = pd.read_excel('../data/BEND_raw.xlsx', header=0) 

ben_data = ben_raw.drop(columns = ['url1', 'url2', 'num_common_ne',
                                   'common_org', 'common_per', 'common_loc'])

ben_texts = pd.read_csv('../data/BEND_texts.csv', header=0)
ben_texts = ben_texts.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
ben_texts.to_csv('../data/BEND_texts.csv', index = True)



# step 2: inspect the data 
def get_mean_length(column1, column2):
    length_per_string = []
    for article in column1:
        length_per_string.append(len(article))
    for article in column2:
        length_per_string.append(len(article))
        
    return(mean(length_per_string))
        
    
summary_length = get_mean_length(ben_data['summary1'], ben_data['summary2'])
title_length = get_mean_length(ben_data['title1'], ben_data['title2'])


# step 3: represent the articles with BoW 

# extract unique titles 
titles = ben_data['title1'].tolist()
titles2 = ben_data['title2'].tolist()
for title in titles2: 
    titles.append(title)

unique_titles = list(set(titles))

# tokenize 
corpus = nltk.sent_tokenize(unique_titles)
print(len(corpus))

# dict of word frequencies 

# create BoW model 


# def get_BoW(df): 
    summaries1 = df['title1']
    summaries2 = df['title2']
    
    
    
    

