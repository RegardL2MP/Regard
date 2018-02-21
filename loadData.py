# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:07:42 2018

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
        
a = loadData("Starting_Kit/sample_data/cifar10_train.data")
print(a.getData())