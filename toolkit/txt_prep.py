#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:18:10 2020

@author: PL - PYC - MH
@version: 1
@description: Fonctions de Traitement du txt (Import/Export/SoumissionKaggle)
"""

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from datetime import datetime
import csv

import nltk
from nltk.corpus import stopwords 
#from nltk.tokenize import word_tokenize
import string
import spacy
nlp = spacy.load("en_core_web_sm")

def deleteStopWords(desc):
    return ' '.join([word for word in desc.split() if word not in ENGLISH_STOP_WORDS])

def cleanText(string: str, punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''', 
               stop_words=['the', 'a', 'and', 'is', 'be', 'will','on'])->str:
    """
    A method to clean text 
    """
    # Cleaning the urls
    string = re.sub(r'https?://\S+|www\.\S+', '', string)

    # Cleaning the html elements
    string = re.sub(r'<.*?>', '', string)

    # Removing the punctuations
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 

    # Converting the text to lower
    string = string.lower()

    # Removing stop words
    string = ' '.join([word for word in string.split() if word not in stop_words])

    # Cleaning the whitespaces
    string = re.sub(r'\s+', ' ', string).strip()

    return string  

def prepareTxt(text, flg_stemm=False, flg_lemm=True, lst_stopwords= set(stopwords.words('english')) ):
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

def prepareTxtSpacy(vectDescription):
    texts = []
    for doc in nlp.pipe(list(vectDescription)):
        doc_lemma = " ".join(token.lemma_.lower() for token in doc if token.lemma_ not in string.punctuation and not token.is_stop)
        texts.append(doc_lemma.strip())
    return texts

def vectTxtSpacy(vectDescription):
    texts = []
    for doc in nlp.pipe(list(vectDescription)):
        doc_vect = [token for token in doc if token.lemma_ not in string.punctuation]
        texts.append(doc_vect)
    return texts






