# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 18:38:05 2020

@author: Marc
"""

# %% Importation

import os
from ManipData.txt_prep import prepareTxt
from ManipData.IO_Kaggle import Import, HistoriqueCsv, macro_disparate_impact

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from dask_ml.model_selection import train_test_split
import dask

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

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

run = wandb.init(name='balance', project="projetsd", config={"preparation":"balanceclass"})
# %% importation données

# Création du dico des `noms de métiers`
jobNames = pd.read_csv('../data/categories_string.csv')['0'].to_dict()

# Importation du jeu de données d'entrainement
trainX=Import(fileName='../data/train',fileExtension='json').set_index('Id')
trainY=pd.read_csv('../data/train_label.csv',index_col='Id')

# Concatenation en un seul DataFrame pour visualiser
trainDF= pd.concat([trainX, trainY], axis=1)
trainDF['description']=trainDF.apply(lambda x:prepareTxt(x['description']),axis=1)
trainSet,validSet=train_test_split(trainDF,shuffle=True,test_size=0.3)


XTrainSet= trainSet.loc[:,trainSet.columns!='Category']
XValidSet= validSet.loc[:,validSet.columns!='Category']


YTrainSet= pd.DataFrame(trainSet['Category'])
YValidSet= pd.DataFrame(validSet['Category'])
#%% function

def balanceClass(resume,data,label):
    """Pour chacun des métiers au dessus de la médiane : 
    1. Selectionner les classes où count > quantile(0.75).
    """
    def selectRows(data,label,classe,thsld):
        df=data.assign(metier=1*(label.Category==classe))
        Y=df.metier
        ## Initialise
        
        tfidf_vect=TfidfVectorizer(min_df=1000)
        clf=SGDClassifier(loss="modified_huber",penalty='l2',early_stopping=True,n_jobs=-1)
        pipeline = Pipeline([('vector',tfidf_vect),("clf",clf)])
        pipeline.fit(df.description,Y)
        pred=pipeline.predict_proba(df.description)
        pred_df=df.copy()
        pred_df['pred'] = pred[:,1]
        pred_df=pred_df.sort_values('pred',ascending=False)

        thsld_G=int(thsld/2)
        
        pred_df_M=pred_df[pred_df.gender=='M'].iloc[:thsld_G]
        pred_df_F=pred_df[pred_df.gender=='F'].iloc[:thsld_G]
        FinalDF=pd.concat([pred_df_M, pred_df_F], ignore_index=True)

        return FinalDF

    thresholdCount=resume['count'].quantile(0.95) ## Nombre de ligne médian parmis toute les classes
    classes=resume[resume['count']>thresholdCount].index.to_list() ## Classes à rééquilibrées
    
    balanced_DF=[selectRows(data,label,classe,thsld=thresholdCount) for classe in classes]                       
    normalizedDF=pd.concat(balanced_DF)
    
    otherclasses=data[~label.Category.isin(classes)]
    classesrow=data[data.index.isin(normalizedDF.index)]
    lastDF=pd.concat([otherclasses,classesrow])
    
    return lastDF

#%%
thsld=4120
resume=trainY.groupby(["Category"]).agg({"Category":["count"]})
resume.columns = resume.columns.droplevel(0)
resume['label']=resume.index.map(jobNames)
new_df=balanceClass(resume,XTrainSet,YTrainSet)

#%%

train_label_reduit=trainY.iloc[new_df.index]

tfidf_vect=TfidfVectorizer(min_df=10,ngram_range=(1,2))
clf=SGDClassifier(loss="modified_huber",penalty='l2',early_stopping=True,n_jobs=-1)

pipeline = Pipeline([('vector',tfidf_vect),("clf",clf)])
#%%


trainDF= pd.concat([new_df, train_label_reduit], axis=1)


#%%

pipeline.fit(trainDF.description, trainDF.Category)

predicted = pipeline.predict(XValidSet.description)

#%%

Macrof1 = f1_score(predicted,YValidSet,average=None)
list_f1 = {x:y for x,y in zip(list(jobNames.values()),Macrof1)}
Macrof1m = f1_score(predicted,YValidSet,average='macro')
#print(f'f1 score =  {Macrof1:.5}')
y_pred = pd.Series(predicted , name='job', index=XValidSet.index)
test_people = pd.concat((y_pred, XValidSet.gender), axis='columns')
fairness = macro_disparate_impact(test_people)
#print(f'Fairness = {fairness:.5}')

run.log({"f1 list": list_f1, "Fairness" : fairness, 'f1 score': Macrof1m})
run.finish()