#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:41:42 2020

@author: pierre
"""

import os

def FolderTree():
    """  Génération de l'arboresence\n
    Returns: arborescence sous forme de texte.
    """    
    return os.system("tree ./")


"cat parts/* > final.md"