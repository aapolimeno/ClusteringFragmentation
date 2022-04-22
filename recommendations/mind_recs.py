# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:19:38 2022

@author: Alessandra
"""

import pickle


path = '../../data/recs_mind/recommendations_large_all_recs_no_cap.pickle'
unpickleFile = open(path, 'rb')
mind_recs = pickle.load(unpickleFile, encoding='latin1')


mind_recs = mind_recs[:100]
