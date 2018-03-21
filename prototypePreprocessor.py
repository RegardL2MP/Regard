# -*- coding: utf-8 -*-
"""
Ceci est reserve au preprocesseurs
"""
from sklearn.base import BaseEstimator 
import numpy as np
import loadData as loadD

moy = []#256 valeurs, chaque moyenne pour 1 feature (colonne)
var = []

def identity(X, num_f): 
    return X 

# Pour nettoyer le dataset des entrées incomplètes, INUTILE toutes les données sont completes
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

def calcul_Moy_Var(l):
    global moy, var    
    l = list(zip(*reversed(l)))
    unpack_Tuple(l)
    
    for i in range(len(l)):
        (v, m) = variance(l[i])
        moy.append(m)
        var.append(v)
            
    l = list(zip(*l)[::-1])
    unpack_Tuple(l)

def moyenne(l):
    s = 0
    for i in range(len(l)):
        s += l[i]
    return s / float(len(l))
    
def variance(l):
    m = moyenne(l)
    s = 0
    for i in range(len(l)):
        s += (l[i] - m)**2
    return ((1 / float(len(l))) * s, m)

def normaliser(l):
    global moy, var
    l = list(zip(*reversed(l)))
    unpack_Tuple(l)
    
    for i in range(len(l)):
        (v, m) = (var[i], moy[i])
        for j in range(len(l)):
            l[i][j] = (l[i][j] - m) / (10**-5 + np.sqrt(v))
            
    l = list(zip(*l)[::-1])
    unpack_Tuple(l)
    return l
    
def normaliser_Test(l, epsilon = 0.00005):
    l = list(zip(*reversed(l)))
    unpack_Tuple(l)
    
    for i in range(len(l)):
        (v, m) = variance(l[i])
        print("ligne [", i, "]: ", l[i])
        print("moy: ", m, " var: ", v)
        assert abs(m) < epsilon
        assert abs(v-1) < epsilon or v < epsilon
            
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

# Permet d'effectuer une analyse en composante principale pour réduire la dimension
def analyse_composante_principale(dataset):
	return PCA(dataset)


class SimpleTransform(BaseEstimator): 
    def fit(self, X, y=None):
        calcul_Moy_Var(X)

    def fit_transform(self, X, y=None): 
        self.fit(X, y)        
        return self.transform(X, y) 

    def transform(self, X, y=None):         
        return normaliser(X)

l = [[1, 0, 3],[4, 0, 7],[5, 0, 9]]
l2 = [1, 2]

#a = loadD.loadData("Starting_Kit/public_data/cifar10_train.data")
#print(a.getData())
#print(moyenne(l2))
#printL(nettoyer_bdd(a.getData(), 256))

normaliser_Test(normaliser(l))