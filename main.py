import numpy as np
import random

from model import *

valuesCol = np.loadtxt('data.txt', usecols=[0,1])
labelsCol = np.loadtxt('data.txt',dtype="str", usecols=[2])

#Removing first row
values = valuesCol[1:]
labels = labelsCol[1:]

#TODO posledna vrstva vzdy o velkosti poctu kategorii, dorobit zautomatizvane
layers = [20,30,3]
(dim, count) = values.shape

#TODO shuffluj data
secDim = count
weights = list()
for j in range(len(layers)):
    weights.append(np.random.normal(0, 1, [layers[j], secDim + 1]))
    secDim = layers[j]

model = Model(weights)
model.train(layers,values,labels,0.1,5)




