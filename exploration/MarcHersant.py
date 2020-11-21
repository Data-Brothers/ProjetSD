# -*- coding: utf-8 -*-
# %%
"""
Created on Wed Oct 21 18:50:13 2020

@author: Marc
"""
# %% Importation

import os
from ManipData.txt_prep import prepareTxt, prepareTxtSpacy
from ManipData.IO_Kaggle import Import, HistoriqueCsv, macro_disparate_impact

import pandas as pd
import re
import nltk
import dask
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD # acp

from sklearn.pipeline import Pipeline
from dask_ml.model_selection import train_test_split
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from sklearn.decomposition import TruncatedSVD


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

# %% wandb
import wandb
from datetime import datetime
jour = str(datetime.today())
wandb.login(key=("8fc6b15f3fe940699f4c8a6a1f7e212afae58c6a"))

# %%
run = wandb.init(name=jour, project="projetsd", config={"preparation":"prepareTxt", 
                                        "transformation":"TfidfVectorizer()",
                                        "modele":'SGDClassifier(loss="modified_huber", penalty="l2",early_stopping=True)',
                                        "Validation": 'Hold-out, test_size=0.3'})

# run = wandb.init(name='nobalance', project="projetsd", config={"preparation":"sans balanceclass"})

# %% importation données

# Création du dico des `noms de métiers`
jobNames = pd.read_csv('./data/categories_string.csv')['0'].to_dict()

# Importation du jeu de données d'entrainement
trainX=Import(fileName='./data/train',fileExtension='json').set_index('Id')
trainY=pd.read_csv('./data/train_label.csv',index_col='Id')

# Concatenation en un seul DataFrame pour visualiser
trainDF= pd.concat([trainX, trainY], axis=1)

trainDF.sort_values(by = 'len_desc',ascending = False).head
#%%
print(trainDF['Category'].value_counts())

#%%
test = trainDF[0:100].copy()
# %%
test['description'] = test.apply(lambda x:prepareTxtSpacy(x['description']),axis=1)
# %% preparation

trainDF['description'] = trainDF.apply(lambda x:prepareTxt(x['description']),axis=1)

# %% modele


# clf=MultinomialNB()
# clf=AdaBoostClassifier(n_estimators=100)

# clf=GradientBoostingClassifier(n_estimators=10)

clf= SGDClassifier(loss="modified_huber", penalty="l2",early_stopping=True)
# clf = RandomForestClassifier(n_jobs=4)



text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf',clf),
])



# %% séparetion train/test

trainDF.rename(columns={'Category': 'Y'}, inplace=True)
trainSet,validSet=train_test_split(trainDF,shuffle=True,test_size=0.3)


XTrainSet= trainSet.loc[:,trainSet.columns!='Y']
XValidSet= validSet.loc[:,validSet.columns!='Y']


YTrainSet= trainSet['Y']
YValidSet= validSet['Y']





# %% entrainement
# Fitting our train data to the pipeline
text_clf.fit(XTrainSet.description , YTrainSet)

predicted = text_clf.predict(XValidSet.description)

# %% score

Macrof1 = f1_score(predicted,YValidSet,average=None)
list_f1 = {x:y for x,y in zip(list(jobNames.values()),Macrof1)}
Macrof1m = f1_score(predicted,YValidSet,average='macro')
#print(f'f1 score =  {Macrof1:.5}')
y_pred = pd.Series(predicted , name='job', index=XValidSet.index)
test_people = pd.concat((y_pred, XValidSet.gender), axis='columns')
fairness = macro_disparate_impact(test_people)
#print(f'Fairness = {fairness:.5}')

# %% log wandb

#run.log({"f1 list": list_f1, "Fairness" : fairness, 'f1 score': Macrof1m})
run.log({"Fairness" : fairness, 'f1 score': Macrof1m})
run.finish()
# %% Exportation
# HistoriqueCsv('Marc', 'prepareTxt, TfidfVectorizer()',
#                 'RandomForest', 
#                 'Hold-out, test_size=0.3', f'{Macrof1:.5}', '{fairness:.5}')

confusion_matrix = multilabel_confusion_matrix(predicted,YValidSet)