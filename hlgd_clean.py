#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:37:45 2022

@author: alessandrapolimeno
"""

import pandas as pd

hlgd_train = pd.read_csv('../../data/hlgd_texts_train.csv', index_col=0)


### Remove recommended titles 

def remove_text_allana(position):
    text = hlgd_train.loc[position, "text"]
    index = text.find('Allana')
    text = text[:index]
    hlgd_train.loc[position,"text"] = text
    print(position)
    print(text)
    print()
    print()

remove_list_a = [903, 842, 801, 767, 746, 737, 719, 712, 687, 559, 
               418, 341, 339, 304, 282, 249, 239, 106, 66, 65, 31, 16]


for number in remove_list_a: 
    remove_text_allana(number)


def remove_text_published(position):
    text = hlgd_train.loc[position, "text"]
    index = text.find('Published')
    text = text[:index]
    hlgd_train.loc[position,"text"] = text
    print(position)
    print(text)
    print()
    print()

remove_list_p = [842, 494]
for number in remove_list_p: 
    remove_text_published(number)
    
    
### Drop CAPTCHA + long
remove_list_c = [623, 579, 572, 569, 465, 374, 351, 346, 338, 307, 293,
                 266, 139, 125, 111, 110, 100, 40, 12, 508, 400, 581, 539]


hlgd_train = hlgd_train.drop(labels = remove_list_c, axis = 0)

hlgd_train.to_csv('../../data/hlgd_text_train2.csv', index = True)
