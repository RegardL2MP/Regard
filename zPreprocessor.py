# -*- coding: utf-8 -*-
"""
Ceci est reserve au preprocesseurs
"""
from sklearn.base import BaseEstimator 
import loadData as loadD

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

# Pour nettoyer le dataset des entrées incomplètes, INUTILE toutes les données sont completes
def nettoyer_bdd(dataset):
    #on fait une copie de la liste pour éviter de modifier la liste de base
        
    c = 0
    cleaned_list = list(dataset)
    for i in range(len(cleaned_list)):
        si il y a moins de 256 features alors le dataset est incomplète
        if(len(cleaned_list[i]) != 256):
            c+=1
            #donc on retire de notre liste l'entrée incriminée
            cleaned_list.pop(i)
    
    rotate = list(zip(*reversed(cleaned_list)))      
    bleached_list = []
    
    for i in range(len(rotate)):
        if(rotate[i] != len(rotate[i]) * [0]):
            bleached_list.append(rotate[i])
            
    bleached_list = list(zip(*bleached_list)[::-1])
    print("removed " + str(len(cleaned_list) - len(bleached_list)) + " columns.")
    return bleached_list

def printL(l):
    s = ""
    for i in range(len(l)):
        for j in range(len(l[0])):
            if(l[i][j] == 0):
                s += "0"
            else:
                s += "X"
        s += "\n"
    return s

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

a = loadD.loadData("Starting_Kit/sample_data/cifar10_train.data")

nettoyer_bdd(a.getData())