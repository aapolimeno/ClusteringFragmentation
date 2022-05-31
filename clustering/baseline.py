# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:22:17 2022

@author: Alessandra
"""
# import dart.handler.elastic.connector
# import dart.models.Handlers
# import dart.handler.NLP.cosine_similarity
# import dart.Util
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from itertools import combinations
import math
import numpy as np

def get_BoW(sentences): 
    CountVec = CountVectorizer(ngram_range=(1,1), stop_words='english')
    Count_data = CountVec.fit_transform(sentences)
    bow_vectors = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())
    
    return bow_vectors


def filter_stopwords(sentence, en):
    filtered_tokens = []
    stopwords = en.Defaults.stop_words
    for token in sentence.split():
        if token.lower() not in stopwords:
            filtered_tokens.append(token)
        
        
    tokens = ' '.join(filtered_tokens)
    
    return tokens


def get_representation(sentences, method = "word"):     
    if method == "SBERT":
        model = SentenceTransformer('all-MiniLM-L6-v2') 
        embeddings = model.encode(sentences)
        
    if method == "word": 
        embeddings = []
        nlp = spacy.load('en_core_web_md')
        for sent in sentences: 
           doc = nlp(sent)
           embeddings.append(doc.vector) 
    
    if method == "BoW": 
        embeddings = get_BoW(sentences)
    
    return embeddings
    


# ================== Get sentence embeddings ==================
df = pd.read_csv("../../data/new_split/new_dev.csv", index_col = 0)
texts = df["text"].tolist()

#test_df = df.iloc[[0, 1, 2, 4, 299, 349]]
#texts = test_df["text"].tolist()
article_pairs = list(combinations(texts, 2))



cosine_df = pd.DataFrame()

cosines = []
indeces = []


# calculate tf_idf


#def get_tfidf_representation(texts):
    
    
corpus = nltk.sent_tokenize(texts)


tfidf_vectorizer = TfidfVectorizer()
# represent texts as tf-idf scores
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
# calculate 
cosine_sim = cosine_similarity(tfidf_matrix)
print(cosine_sim)


# approach 2 

def cosine_similarity_calc(vec_1,vec_2):
	
	sim = np.dot(vec_1,vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2))
	
	return sim


embeddings = get_representation(texts, "word")
article_pairs = list(combinations(embeddings, 2))

for pair in article_pairs: 
    emb1 = pair[0]
    emb2 = pair[1]
    
    cosine_sim = cosine_similarity_calc(emb1, emb2)
    cosines.append(cosine_sim)

# Torch 

embeddings = get_representation(texts, "word")

final = []
for i in range(len(embeddings)):
  ans = []
  for j in range(len(embeddings)):
  	similarity = cosine_similarity(embeddings[i].view(1,-1), 
  										 embeddings[j].view(1,-1)).item()
  	ans.append(similarity)
  final.append(ans)




#cosine = cosine_similarity(emb_text1, emb_text2)

##cosine = cosine(emb_text1, emb_text2)
##if cosine > 0.5:
#cosines.append(cosine)
    
#cosine_df["index_pairs"] = indeces
cosine_df["cosine_similarity"] = cosines




# ================== Step 1: calculate cosine similarities for article pairs ==================
# cosine = cosine_similarity(emb_text1, emb_text2)



