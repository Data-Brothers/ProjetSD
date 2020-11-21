# -*- coding: utf-8 -*-

from ManipData.txt_prep import vectTxtSpacy, prepareTxtSpacy, prepareTxt
from ManipData.IO_Kaggle import Import, macro_disparate_impact

import pandas as pd
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")

from sklearn.linear_model import SGDClassifier
from dask_ml.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# %% importation données

# Création du dico des `noms de métiers`
jobNames = pd.read_csv('./data/categories_string.csv')['0'].to_dict()

# Importation du jeu de données d'entrainement
trainX=Import(fileName='./data/train',fileExtension='json').set_index('Id')
trainY=pd.read_csv('./data/train_label.csv',index_col='Id')

# Concatenation en un seul DataFrame pour visualiser
trainDF= pd.concat([trainX, trainY], axis=1)

#%%
test = trainDF[0:100].copy()
# %%
test['description'] = test.apply(lambda x:prepareTxtSpacy(x['description']),axis=1)
#%%
trainDF.to_csv('spacyembedding_nostop.csv')

#%%
clf= SGDClassifier(loss="modified_huber", penalty="l2",early_stopping=True)

text_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',clf),
])
# %% séparetion train/test

trainDF.rename(columns={'Category': 'Y'}, inplace=True)
trainSet,validSet=train_test_split(trainDF,shuffle=True,test_size=0.3)


XTrainSet= trainSet.loc[:,trainSet.columns!='Y']
XValidSet= validSet.loc[:,validSet.columns!='Y']


YTrainSet= trainSet['Y']
YValidSet= validSet['Y']


#%%
XTrain = np.array(XTrainSet.description)
XValid = np.array(XValidSet.description)

#%%
text_clf.fit(list(XTrain) , YTrainSet)

predicted = text_clf.predict(list(XValid))

#%%

Macrof1 = f1_score(predicted,YValidSet,average=None)
list_f1 = {x:y for x,y in zip(list(jobNames.values()),Macrof1)}
Macrof1m = f1_score(predicted,YValidSet,average='macro')
print(f'f1 score =  {Macrof1:.5}')
y_pred = pd.Series(predicted , name='job', index=XValidSet.index)
test_people = pd.concat((y_pred, XValidSet.gender), axis='columns')
fairness = macro_disparate_impact(test_people)
print(f'Fairness = {fairness:.5}')
