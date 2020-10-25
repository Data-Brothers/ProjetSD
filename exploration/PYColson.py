#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18

@author: Pierre-Yves COLSON
"""
import os
import sys
sys.path.append('../')
from sklearn import model_selection


from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, LassoCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ManipData.txt_prep import cleanText,deleteStopWords, prepareTxt, prepareTxtSpacy
from ManipData.IO_Kaggle import macro_disparate_impact, Import, Export

import pandas as pd
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
from dask_ml.model_selection import train_test_split





# Nom des jobs

# names = pd.read_csv('../data/categories_string.csv')['0'].to_dict()

# data=Import(fileName='../data/train',fileExtension='json').set_index('Id')

# train_label=pd.read_csv('./data/train_label.csv',index_col='Id')

# data_test=Import(fileName='../data/test',fileExtension='json').set_index('Id')



#############################################################################
#####                 Statistiques descriptives                         #####
#############################################################################


# description = pd.concat([train_label.Category, data.gender], axis = 1)
# description.gender[description.gender=="F"] = 1
# description.gender[description.gender=="M"] = 0

# resume = pd.DataFrame({"label" : list(range(28)),
#                       "femmes" : description.groupby("Category").sum().gender, 
#                       "total" : description.groupby("Category").count().gender})

# for i in range(len(resume)):
#     resume.label[i] = names[resume.label[i]]

# # Graph nb total d'individus par profession

# resume.sort_values('total',inplace=True)
# resume.plot(kind='barh',y='total',x='label',color='r')

# # Graph % de femmes par profession

# resume.femmes = list(map(lambda x, y: x/y*100, resume.femmes, resume.total))
# resume.sort_values('femmes',inplace=True)
# resume.plot(kind='barh',y='femmes',x='label',color='r')


#############################################################################
#####                          ML Algos                                 #####
#############################################################################



#%% importation données

# Création du dico des `noms de métiers`
jobNames = pd.read_csv('../data/categories_string.csv')['0'].to_dict()


# Importation du jeu de données d'entrainement

trainX=Import(fileName='../data/train',fileExtension='json').set_index('Id')

trainY=pd.read_csv('../data/train_label.csv',index_col='Id')

# Concatenation en un seul DataFrame pour visualiser
trainDF= pd.concat([trainX, trainY], axis=1)

#%% preparation
# utilisation de la fonction prepareTxt
#trainDF['description']=trainDF.apply(lambda x:prepareTxt(x['description']),axis=1)

# utilisation de la fonction prepareTxt
trainDF['description'] = prepareTxtSpacy(trainDF['description'])

#%% modele

# clf=LinearDiscriminantAnalysis()
# clf=MultinomialNB()
# clf=AdaBoostClassifier(n_estimators=100)

# clf=GradientBoostingClassifier(n_estimators=10)

#clf= SGDClassifier(loss="modified_huber", penalty="l2",early_stopping=True)
#clf = RandomForestClassifier(n_jobs=4)

cv_outer = KFold(len(trainDF), n_folds=5)
clf = LassoCV(cv=3)  # cv=3 makes a KFold inner splitting with 3 folds

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

#scores = cross_val_score(lasso, X, y, cv=cv_outer)


