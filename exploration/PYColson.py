#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18

@author: Pierre-Yves COLSON
"""

from sklearn import model_selection

from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics
from manipData import Import,Export,cleanText,deleteStopWords
import pandas as pd
from pandas import DataFrame, Series
from nltk.stem import PorterStemmer 
from collections  import defaultdict

import dask
import dask.multiprocessing
from dask import delayed
import dask.dataframe as dd
import matplotlib.pyplot as plt
import string




# Nom des jobs
names = pd.read_csv('./data/categories_string.csv')['0'].to_dict()

data=Import(fileName='./data/train',fileExtension='json').set_index('Id')
train_label=pd.read_csv('./data/train_label.csv',index_col='Id')

data_test=Import(fileName='./data/test',fileExtension='json').set_index('Id')


#############################################################################
#####                 Statistiques descriptives                         #####
#############################################################################


description = pd.concat([train_label.Category, data.gender], axis = 1)
description.gender[description.gender=="F"] = 1
description.gender[description.gender=="M"] = 0

resume = pd.DataFrame({"label" : list(range(28)),
                      "femmes" : description.groupby("Category").sum().gender, 
                      "total" : description.groupby("Category").count().gender})

for i in range(len(resume)):
    resume.label[i] = names[resume.label[i]]

# Graph nb total d'individus par profession

resume.sort_values('total',inplace=True)
resume.plot(kind='barh',y='total',x='label',color='r')

# Graph % de femmes par profession

resume.femmes = list(map(lambda x, y: x/y*100, resume.femmes, resume.total))
resume.sort_values('femmes',inplace=True)
resume.plot(kind='barh',y='femmes',x='label',color='r')


# essais diverses

post1 = data.description[1]
print(post1)
# post1 = deleteStopWords(post1)
# print(post1)
# post1 = cleanText(post1)
# print(post1)

# words = post1.lower().split()


# stemmer = PorterStemmer()
# wordStem = []
# for word in words :
#     wordStem.append(stemmer.stem(word))




# dataset = []
# doc = defaultdict(int)
# for word in wordStem : 
#     doc[word] +=1
# dataset.append(doc)
    

def wordFreq(doc):
    post = cleanText(deleteStopWords(doc))
    words = post.lower().split()
    stemmer = PorterStemmer()
    wordStem = [stemmer.stem(word) for word in words]
    doc_corr = defaultdict(int)
    for word in wordStem : 
        doc_corr[word] +=1
    valMax = max(doc_corr.values())
    return {k:v/valMax for k,v in doc_corr.items()}

print(wordFreq(data.description[1]))


# tasks = [delayed(wordFreq(data.description[i])) for i in range(10000)]
# list_df = tasks.compute()
# tasks.visualize()
#dask.compute(*tasks)
#data_train = DataFrame(list_df)

%%time
L = [wordFreq(data.description[i]) for i in range(5000)]
train = DataFrame(L)

%%time
L = [dask.delayed(wordFreq)(data.description[i]) for i in range(5000)]
L = dask.compute(L)[0]
train = DataFrame(L)




import spacy
nlp = spacy.load("en_core_web_sm")

def prepareTxtSpacy(text):
    doc = nlp(text.strip())
    doc_lemma = " ".join([token.lemma_.lower() for token in doc if token.lemma_ not in string.punctuation and not token.is_stop])
    return doc_lemma



