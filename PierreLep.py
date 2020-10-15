#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:59:13 2020

@author: PierreLepagnol
"""

from sklearn import model_selection

from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics
from manipData import Import,Export,cleanText,deleteStopWords
import pandas as pd


# Nom des jobs
names = pd.read_csv('./data/categories_string.csv')['0'].to_dict()

data=Import(fileName='./data/train',fileExtension='json').set_index('Id')
train_label=pd.read_csv('./data/train_label.csv',index_col='Id')

data_test=Import(fileName='./data/test',fileExtension='json').set_index('Id')
# X_train=data["description"].apply(deleteStopWords)
# Y_train=train_label

# DF= X_train.map(names)

# model = pipeline.make_pipeline(
#     feature_extraction.text.TfidfVectorizer(),
#     preprocessing.Normalizer(),
#     linear_model.LogisticRegression(multi_class='multinomial')
# )

# model = model.fit(X_train, Y_train)

# y_pred = model.predict(X_test)
# y_pred = pd.Series(y_pred, name='job', index=X_test.index)

# print(metrics.f1_score(y_test, y_pred, average='macro'))

# # dtype={'Id':'float','description':'string','gender':'category'}
# # Train_X=Import(fileName='./data/train',fileExtension='json',dtype=dtype)
# # # Train_Label=Import(fileName='./data/train_label',fileExtension='csv')

# # # Test_X=Import(fileName='./data/test',fileExtension='json')


# # # Train_X = Train_X.set_index('Id')
# # res=Train_X.apply(lambda x : cleanText(x['description']),axis=1)




# test_people = pd.concat((y_pred, gender_test), axis='columns')

# macro_disparate_impact(test_people)
