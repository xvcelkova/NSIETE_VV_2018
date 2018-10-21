import numpy as np
import random

from model import *

valuesCol = np.loadtxt('data.txt', usecols=[0,1])
labelsCol = np.loadtxt('data.txt',dtype="str", usecols=[2])

#Removing first row
values = valuesCol[1:]
labels = labelsCol[1:]

values[:,0] -= np.mean(values[:,0])
values[:,0] /= np.std(values[:,0])
values[:,1] -= np.mean(values[:,1])
values[:,1] /= np.std(values[:,1])

#TODO posledna vrstva vzdy o velkosti poctu kategorii, dorobit zautomatizvane
layers = [20,10,3]
(dim, count) = values.shape

#TODO shuffluj data
secDim = count
weights = list()
for j in range(len(layers)):
    weights.append(np.random.normal(0, 1, [layers[j], secDim + 1]))
    secDim = layers[j]

model = Model(weights)
model.train(layers,values,labels,0.1,100)




