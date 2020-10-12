#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:18:10 2020

@author: PL - PYC - MH
@version: 1
@description: Fonctions de manipulation des données (Import/Export/SoumissionKaggle)
"""

import os
import pandas as pd

def KaggleSubmit(fileName,Msg):
    """  Soumission sur kaggle\n
    Params:
    fileName (str) Nom du fichieravec l'extension\t
    Msg (str) Message à afficher pour la soumission
    
    Returns: 0 (Everything runned well)
    """
    cmdSubmit="kaggle competitions submit -c defi-ia-insa-toulouse -f {fName} -m '{Msg}'".format(Msg=Msg,fName=fileName)
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
    cmdread = f'pd.read_{fileExtension}(f)'
    with open(fileName_Ext,'r') as f:
        dataset=eval(cmdread) 
    print(cmdread)
    return dataset

def Export(pdDF=pd.DataFrame(),fileName='./data/unknown',fileExtension='csv',KaggleSubmission=False,MsgSubmit='New Submission from DataBrothers'):
    fileName_Ext=f"{fileName}.{fileExtension}"
    df.to_csv(fileName_Ext, index = False, header=True)
    if(KaggleSubmission): KaggleSubmit(fileName_Ext,MsgSubmit)
    return 0
