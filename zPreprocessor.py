# -*- coding: utf-8 -*-
"""
Ceci est reserve au preprocesseurs
"""
from sklearn.base import BaseEstimator 
import loadData.py

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

# Pour nettoyer le dataset des entrées incomplètes
def nettoyer_bdd(dataset):
    #on fait une copie de la liste pour éviter de modifier la liste de base
    cleaned_list = list(dataset)
    for i in range(len(cleaned_list)):
        #si il y a moins de 256 features alors le dataset est incomplète
        if(len(cleaned_list[i]) != 256):
            #donc on retire de notre liste l'entrée incriminée
            cleaned_list.pop(i)
    #puis on renvoie la liste nettoyée
    return cleaned_list

# Permet de normaliser les valeurs du dataset
# mini/maxi = valeur minimale/maximale qu'on souhaite obtenir pour chaque valeur
def normaliser_bdd(dataset, mini, maxi):
    if(mini == maxi or maxi < mini):
        raise ValueError("valeurs mini maxi fausses")
    normalized_list = list(dataset)    
    #pour chaque entrée
    for i in range(len(normalized_list)):
        for j in range(len(normalized_list[i])):
            #formule de la normalisation
            normalized_list[i][j] = (normalized_list[i][j] - mini) / (maxi - mini)
    return normalized_list
