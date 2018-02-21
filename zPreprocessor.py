# -*- coding: utf-8 -*-
"""
Ceci est reserve au preprocesseurs
"""

from sklearn.base import BaseEstimator

def idd(x):
    return x

class PreProcess(BaseEstimator):
    def __init__(self, transform=idd):
        self.transform = idd
        
    #rajouter les methodes pour nettoyer les donnees
        