#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:23:51 2020

@author: pierre-yvescolson
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:50:13 2020

@author: Marc
"""
#%% Importation

import os
from manipData import Import, HistoriqueCsv, macro_disparate_impact
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from dask_ml.model_selection import train_test_split

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

#%% importation données

# Création du dico des `noms de métiers`
jobNames = pd.read_csv('./data/categories_string.csv')['0'].to_dict()


# Importation du jeu de données d'entrainement

trainX=Import(fileName='./data/train',fileExtension='json').set_index('Id')

trainY=pd.read_csv('./data/train_label.csv',index_col='Id')

# Concatenation en un seul DataFrame pour visualiser
trainDF= pd.concat([trainX, trainY], axis=1)

#%% preparation

trainDF['description']=trainDF.apply(lambda x:prepareTxt(x['description']),axis=1)


#%% modele

# clf=LinearDiscriminantAnalysis()
# clf=MultinomialNB()
# clf=AdaBoostClassifier(n_estimators=100)

# clf=GradientBoostingClassifier(n_estimators=10)

# clf= SGDClassifier(loss="modified_huber", penalty="l2",early_stopping=True)
clf = RandomForestClassifier(n_jobs=4)

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf',clf),
])

#%% séparetion train/test

trainDF.rename(columns={'Category': 'Y'}, inplace=True)
trainSet,validSet=train_test_split(trainDF,shuffle=True,test_size=0.3)


XTrainSet= trainSet.loc[:,trainSet.columns!='Y']
XValidSet= validSet.loc[:,validSet.columns!='Y']


YTrainSet= trainSet['Y']

YValidSet= validSet['Y']

#%% entrainement
# Fitting our train data to the pipeline
text_clf.fit(XTrainSet.description, YTrainSet)

predicted = text_clf.predict(XValidSet.description)

#%% score
from sklearn.metrics import f1_score
Macrof1 = f1_score(predicted,YValidSet,average='macro')
print(f'f1 score =  {Macrof1:.5}')
y_pred = pd.Series(predicted , name='job', index=XValidSet.index)
test_people = pd.concat((y_pred, XValidSet.gender), axis='columns')
fairness = macro_disparate_impact(test_people)
print(f'Fairness = {fairness:.5}')

#%% Exportation
HistoriqueCsv('Marc', 'prepareTxt, TfidfVectorizer()',
                'RandomForest', 
                'Hold-out, test_size=0.3', Macrof1, fairness)

