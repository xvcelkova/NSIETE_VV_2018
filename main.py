import numpy as np
import random
import matplotlib.pyplot as plt

from model import *
from util import *

#load data
valuesCol = np.loadtxt('data.txt', usecols=[0,1])
labelsCol = np.loadtxt('data.txt',dtype="str", usecols=[2])
testDataValuesCol = np.loadtxt('test_data.txt', usecols=[0,1])
testDataLabelsCol = np.loadtxt('test_data.txt',dtype="str", usecols=[2])

#Removing first row
values = valuesCol[1:]
labels = labelsCol[1:]
testDataValues = testDataValuesCol[1:]
testDataLabels = testDataLabelsCol[1:]

#normalization
values[:,0] -= np.mean(values[:,0])
values[:,0] /= np.std(values[:,0])
values[:,1] -= np.mean(values[:,1])
values[:,1] /= np.std(values[:,1])

testDataValues[:,0] -= np.mean(testDataValues[:,0])
testDataValues[:,0] /= np.std(testDataValues[:,0])
testDataValues[:,1] -= np.mean(testDataValues[:,1])
testDataValues[:,1] /= np.std(testDataValues[:,1])

#shuffle data
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

# sigmoid/softmax/tanh
functions = ["tanh","tanh","softmax"]
(dim, count) = values.shape

secDim = count
weights = list()
for j in range(len(layers)):
    weights.append(np.random.normal(0, 1, [layers[j], secDim + 1]))
    secDim = layers[j]

model = Model(weights,layers,functions)
model.train(train_inputs,train_labels,0.01,90)
CE, RE, random, predicted_labels = model.test(test_inputs,test_labels)
print('Validation error: CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))

CE, RE, random, predicted_labels = model.test(testDataValues,testDataLabels)
print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))
plt.scatter(testDataValues[:,0],testDataValues[:,1],c = encode_color(testDataLabels))
plt.show()

plt.scatter(testDataValues[:,0][random],testDataValues[:,1][random],c = encode_color(predicted_labels))
plt.show()


