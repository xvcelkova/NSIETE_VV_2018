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

ind = np.arange(0,len(values[:,0]))
random.shuffle(ind)
split =  int(len(values[:,0])* 0.8)
train_ind = ind[:split]
test_ind  =  ind[split:]

train_inputs = values[train_ind,:]
train_labels = labels[train_ind]

test_inputs =  values[test_ind,:]
test_labels =  labels[test_ind]

#TODO posledna vrstva vzdy o velkosti poctu kategorii, dorobit zautomatizvane
layers = [20,10,3]

# sigmoid/softmax
functions = ["sigmoid","sigmoid","sigmoid"]
(dim, count) = values.shape

secDim = count
weights = list()
for j in range(len(layers)):
    weights.append(np.random.normal(0, 1, [layers[j], secDim + 1]))
    secDim = layers[j]

model = Model(weights,layers,functions)
model.train(train_inputs,train_labels,0.1,20)
CE, RE = model.test(test_inputs,test_labels)
print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))

