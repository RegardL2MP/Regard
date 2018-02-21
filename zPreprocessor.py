# -*- coding: utf-8 -*-
"""
Ceci est reserve au preprocesseurs
"""
from sklearn.base import BaseEstimator 

def identity(x): 

    return x 

class SimpleTransform(BaseEstimator): 

    def __init__(self, transformer=identity): 

        self.transformer = transformer 

    def fit(self, X, y=None): 

        return self 

    def fit_transform(self, X, y=None): 

        return self.transform(X) 

    def transform(self, X, y=None): 

        return np.array([self.transformer(x) for x 

in X], ndmin=2).T 
