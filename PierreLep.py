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

