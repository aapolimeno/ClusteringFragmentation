# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:30:14 2022

@author: Alessandra
"""

import pandas as pd 


def convert_labels(convert_dict, gold_labels):
    converted_labels = []
    for label in gold_labels:
        label = convert_dict[label]
        converted_labels.append(label)
    
    return converted_labels


# ================= Validation data =================
# Load full train data (needed for gold labels)
hlgd_texts_dev = pd.read_csv('../../data/hlgd_dev_full.csv', index_col=0)

# Extract all urls and gold labels as a dictionary 
# key: url, value: gold label
dev_url_a = dict(zip(hlgd_texts_dev.url_a, hlgd_texts_dev.timeline_id))
dev_url_b = dict(zip(hlgd_texts_dev.url_b, hlgd_texts_dev.timeline_id))

# Filter duplicats
dev_urls = []
dev_labels = []

for k, v in dev_url_a.items(): 
    dev_urls.append(k)
    dev_labels.append(v)


for k, v in dev_url_b.items(): 
    if k not in dev_urls:
        dev_urls.append(k)
        dev_labels.append(v)
        
dev_label = pd.DataFrame()
dev_label['url'] = dev_urls
dev_label['gold_label'] = dev_labels

# ====== Convert labels ======
convert_dev = {0:0, 5:1}
dev_conv_labels = dev_label['gold_label'].tolist()
new_dev_labels = convert_labels(convert_dev, dev_conv_labels)

dev_label['gold_label'] = new_dev_labels

# Save 
dev_label.to_csv('../../data/gold_dev.csv', index = True)


# ================= Training data =================
# Load full train data (needed for gold labels)
hlgd_texts_full = pd.read_csv('../../data/hlgd_train_full.csv', index_col=0)

# Extract all urls and gold labels as a dictionary 
# key: url, value: gold label
gold_url_a = dict(zip(hlgd_texts_full.url_a, hlgd_texts_full.timeline_id))
gold_url_b = dict(zip(hlgd_texts_full.url_b, hlgd_texts_full.timeline_id))

# Filter duplicats
gold_urls = []
gold_labels = []

for k, v in gold_url_a.items(): 
    gold_urls.append(k)
    gold_labels.append(v)


for k, v in gold_url_b.items(): 
    if k not in gold_urls:
        gold_urls.append(k)
        gold_labels.append(v)
        
 
gold_label = pd.DataFrame()
gold_label['url'] = gold_urls
gold_label['gold_label'] = gold_labels

# ================= Test data =================

# Load full train data (needed for gold labels)
hlgd_texts_full = pd.read_csv('../../data/hlgd_test_full.csv', index_col=0)

# Extract all urls and gold labels as a dictionary 
# key: url, value: gold label
test_url_a = dict(zip(hlgd_texts_full.url_a, hlgd_texts_full.timeline_id))
test_url_b = dict(zip(hlgd_texts_full.url_b, hlgd_texts_full.timeline_id))

# Filter duplicats
test_urls = []
test_labels = []

for k, v in test_url_a.items(): 
    test_urls.append(k)
    test_labels.append(v)


for k, v in test_url_b.items(): 
    if k not in test_urls:
        test_urls.append(k)
        test_labels.append(v)
                

test_label = pd.DataFrame()
test_label["url"] = test_urls
test_label["gold_label"] = test_labels


# ================= Merge train + test ================= 
gold_merge = gold_label.append(test_label, ignore_index = True)
gold_range = gold_merge["gold_label"].tolist()
unique = set(gold_range)

# ====== Convert labels ====== 
convert_dict = {1:5, 2:2, 3:3, 4:4, 6:6, 7:7, 8:8, 9:9}
new_labels = convert_labels(convert_dict, gold_range)

gold_merge['gold_label'] = new_labels

# ================= Save ================= 
gold_merge.to_csv('../../data/gold_labels.csv', index = True)



# ================= Explore ================= 
c0 =  dev_label.loc[dev_label['gold_label'] == 0]
c1 =  dev_label.loc[dev_label['gold_label'] == 1]
c2 =  gold_merge.loc[gold_merge['gold_label'] == 2]
c3 =  gold_merge.loc[gold_merge['gold_label'] == 3]
c4 =  gold_merge.loc[gold_merge['gold_label'] == 4]
c5 =  gold_merge.loc[gold_merge['gold_label'] == 5]
c6 =  gold_merge.loc[gold_merge['gold_label'] == 6]
c7 =  gold_merge.loc[gold_merge['gold_label'] == 7]
c8 =  gold_merge.loc[gold_merge['gold_label'] == 8]
c9 =  gold_merge.loc[gold_merge['gold_label'] == 9]

