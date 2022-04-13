# ClusteringFragmentation

This repository contains the following scripts: 
- hlgd_extraction.py To scrape articles from the URL present in the HLGD dataset
- hlgd_clean.py To filter out irrelevant articles 
- clustering.py To perform Hierarchical Clustering with Bag of Words representations of the articles 



Step 1: Extract and preprocess data 
- Obtain and clean the HLGD data set containing train, test and dev data
a) Scrape the articles of train/test/dev with hlgd_extraction.py
b) Filter irrelevant/incorrect articles for train/test/dev sets with their corresponding hlgd_clean.py script
c) Merge train and test with merge_train_test.py


Step 2: Perform clustering 
a) 


Step 3: Generate recommendations 
- For random recommendations, see random_recs.py