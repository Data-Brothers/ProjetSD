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

#%% fonction
'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def prepareTxt(text, flg_stemm=False, flg_lemm=True, lst_stopwords= set(stopwords.words('english')) ):
    ## clean (convert to lowercase and remove punctuations and   
    ##characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    
    liste_regex=[(r"dr\.","doctor"),(r'(she|he)','')]
    for regex,new in liste_regex: 
        text=re.sub(regex,new,text)
        
    ## Tokenize (convert from string to list)
    lst_text = text.split()    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]            
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text


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

