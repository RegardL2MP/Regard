# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:07:42 2018
NE PAS SUPPRIMER
@author: theo.bourdon
"""

class loadData:
    
    def __init__(self, filename):
        self.content = []
        f = open(filename, 'r ')
        for i in f:
            line = i.strip().split()
            temp = []
            for j in range(0, len(line)):
                temp.append(float(line[j]))
            self.content.append(temp)
    
    def getData(self):
        return self.content            

