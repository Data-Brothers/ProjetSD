#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:18:10 2020

@author: PL - PYC - MH
@version: 1
@description: Fonctions de manipulation des données (Import/Export/SoumissionKaggle)
"""

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score
from datetime import datetime
import csv

def KaggleSubmit(fileName,Msg):
    """  Soumission sur kaggle\n
    Params:
    fileName (str) Nom du fichieravec l'extension\t
    Msg (str) Message à afficher pour la soumission
    
    Returns: 0 (Everything runned well)
    """
    cmdSubmit=f"kaggle competitions submit -c defi-ia-insa-toulouse -f {fName} -m '{Msg}'".format(Msg=Msg,fName=fileName)
    os.system(cmdSubmit)
    cmdLeaderBoard='kaggle competitions leaderboard -c defi-ia-insa-toulouse --show'
    os.system(cmdLeaderBoard)    
    return 0

def Import(fileName='unknown',fileExtension='csv'):
    """ Importation des données\n
    Params:
    fileName (str) Nom du fichier\t
    fileExtension (str) Extension du fichier

    Returns: dataset: DataFrame Pandas
    """
    fileName_Ext=f"{fileName}.{fileExtension}"
    cmdread = f"pd.read_{fileExtension}('{fileName_Ext}')"
    dataset=eval(cmdread)  
    return dataset

def Export(pdDF=pd.DataFrame(),fileName='./data/unknown',fileExtension='csv',
           KaggleSubmission=False,MsgSubmit='New Submission from DataBrothers'):
    fileName_Ext=f"{fileName}.{fileExtension}"
    
    cmdto = f"pdDF.to_{fileExtension}('{fileName_Ext}', index = False, Header=True)"
    eval(cmdto)
    if(KaggleSubmission): KaggleSubmit(fileName_Ext,MsgSubmit)
    return 0

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

def macro_disparate_impact(people):
    counts = people.groupby(['job', 'gender']).size().unstack('gender')
    counts['disparate_impact'] = counts[['M', 'F']].max(axis='columns') / counts[['M', 'F']].min(axis='columns')
    return counts['disparate_impact'].mean()

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def HistoriqueCsv(nom, preparation, modele, validation, f1score, fairness):
    '''
    Parameters
    ----------
    nom : str
        Ton nom.
    preparation : str
        Méthode de préparation de text utilisé.
    modele : str
        Modèle utilisé (obliez pas les parametres).
    validation : str
        Validation effectuée (hold-out, VC...).
    f1score : float
        macro f1-score.
    fairness : float
        macro_disparate_impact.
    -------
    Ecrit le résultat d'un modèle dans le fichier Historique_resultats, 
    pour pouvoir garder une trace des modèles testés.

    '''
    jour = datetime.today()
    with open('Historique_resultats.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow([jour, nom, preparation, modele, validation, f1score, fairness])
