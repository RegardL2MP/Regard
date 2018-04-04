#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: isabelleguyon

This is an example of program that preprocesses data.
It calls the PCA function from scikit-learn.
Replace it with programs that:
    normalize data (for instance subtract the mean and divide by the standard deviation of each column)
    construct features (for instance add new columns with products of pairs of features)
    select features (see many methods in scikit-learn)
    perform other types of dimensionality reductions than PCA
    remove outliers (examples far from the median or the mean; can only be done in training data)
"""

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
import numpy as np

# Pour nettoyer le dataset des entrées incomplètes, INUTILE car toutes les données sont completes
# Cette fonction n'est donc pas utilise au final
def nettoyer_bdd(dataset, num_f):
    #on fait une copie de la liste pour éviter de modifier la liste de base
        
    c = 0
    cleaned_list = list(dataset)
    print(len(cleaned_list), len(dataset))
    i = 0
    while i < len(cleaned_list):
        #si il y a moins de 256 features alors le dataset est incomplète
        if(len(cleaned_list[i]) != num_f):
            c+=1
            #donc on retire de notre liste l'entrée incriminée
            cleaned_list.pop(i)
            i -= 1
        i += 1
    
    #tourne la liste, plus facile pour verifier la présence de colonnes de zeros
    #retourne une liste de tuple; du coup on les transforme en listes avec unpack_Tuple
    rotate = list(zip(*reversed(cleaned_list)))
    unpack_Tuple(rotate)
    print(rotate)
        
    
    bleached_list = []    
    
    for i in range(len(rotate)):
        if(rotate[i] != len(rotate[i]) * [0]):
            bleached_list.append(rotate[i])
            
    print(bleached_list)
    
    bleached_list = list(zip(*bleached_list)[::-1])
    print("removed " + str(len(cleaned_list) - len(bleached_list)) + " columns.")
    unpack_Tuple(bleached_list)    
    
    return bleached_list

def unpack_Tuple(l):
    for i in range(len(l)):
        l[i] = list(l[i])


# Ce preprocessor normalise les valeurs du dataset, en utilisant la moyenne et la deviation standard
class NormalizePreprocessor(BaseEstimator):
    def __init__(self):
        '''
        This example does not have comments: you must add some.
        Add also some defensive programming code, like the (calculated) 
        dimensions of the transformed X matrix.
        '''
        # self.transformer = PCA(n_components=10)
        # print("PREPROCESSOR=" + self.transformer.__str__())
        self.moyennes = None
        self.stds = None

    def calculer_moyenne_std(self, dataset):
        self.moyennes = np.mean(dataset, axis=0)
        self.stds = np.std(dataset, axis=0)
    
    def normaliser_bdd(self, dataset):
        return (dataset - self.moyennes) / (1e-5 + self.stds)

    def fit(self, X, y=None):
        print("PREPRO FIT")
        self.calculer_moyenne_std(X)
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def transform(self, X, y=None):
        print("PREPRO TRANSFORM")
        return self.normaliser_bdd(X)

class PCAPreprocessor(BaseEstimator):
    def __init__(self):
        '''
        This example does not have comments: you must add some.
        Add also some defensive programming code, like the (calculated) 
        dimensions of the transformed X matrix.
        '''
        self.transformer = PCA(n_components=192)
        print("PREPROCESSOR=" + self.transformer.__str__())

    def fit(self, X, y=None):
        print("PREPRO FIT PCA")
        return self.transformer.fit(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        print("PREPRO TRANSFORM PCA")
        return self.transformer.transform(X)
    
if __name__=="__main__":
    # Put here your OWN test code
    
    # To make sure this runs on Codalab, put here things that will not be executed on Codalab
    from sys import argv, path
    path.append ("../starting_kit/ingestion_program") # Contains libraries you will need
    from data_manager import DataManager  # such as DataManager
    
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data" # Replace by correct path
        output_dir = "../results" # Create this directory if it does not exist
    else:
        input_dir = argv[1]
        output_dir = argv[2];
    
    basename = 'cifar10'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print D
    
    Prepro = PCAPreprocessor()
 
    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train']) # Les donnes sont bien de dimensions 192
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
  
    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print D
    
