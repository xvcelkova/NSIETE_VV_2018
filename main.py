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
validation_ind  =  ind[split:]

train_inputs = values[train_ind,:]
train_labels = labels[train_ind]
validation_inputs =  values[validation_ind,:]
validation_labels =  labels[validation_ind]

#models architecture
layers = list()
layers1 = [128,128,128,3]
layers2 = [128,3]
layers3 = [30,20,3]
layers4 = [70,50,40,3]
layers.append(layers1)
#layers.append(layers2)
#layers.append(layers3)
#layers.append(layers4)


# sigmoid/softmax/tanh
functions = list()
functions1 = ["tanh","tanh","tanh","softmax"]
functions2 = ["tanh","softmax"]
functions3 = ["tanh","tanh","softmax"]
functions4 = ["tanh","tanh","tanh","softmax"]
functions.append(functions1)
#functions.append(functions2)
#functions.append(functions3)
#functions.append(functions4)


(dim, count) = values.shape

#weight martices
all_weights = list()
for i in range(len(layers)):
    secDim = count
    weights = list()
    for j in range(len(layers[i])):
        weights.append(np.random.normal(0, np.sqrt(2 / secDim), [layers[i][j], secDim + 1]))
        secDim = layers[i][j]
    all_weights.append(weights)

#train data + validation
validation_errors = list()
epochs = list()
for i in range(len(layers)):
    print("Model ", i+1)
    model = Model(all_weights[i],layers[i],functions[i])
    train_CE, train_RE, train_CEs, train_REs, ep, val_CE, val_RE = model.train(train_inputs,train_labels,0.1,150,validation_inputs,validation_labels)
    epochs.append(ep)
    validation_errors.append(train_CE)
    print("LAST EPOCH")
    print(ep, 'epoch, CE = {:6.2%}, RE = {:.5f}'.format(train_CE, train_RE))
    print('Validation error: CE = {:6.2%}, RE = {:.5f}'.format(val_CE, val_RE))
    print("------------------------------------------")

#choose best model according to minimal validation error
minimum = float("inf")
index = len(layers)+1
for i in range(len(layers)):
    if(validation_errors[i]<minimum):
        minimum = validation_errors[i]
        index = i

#train on whole set with best model
print("BEST MODEL - architecture ",layers[index])
model = Model(all_weights[index],layers[index],functions[index])
train_CE, train_RE, train_CEs, train_REs, ep = model.train(values,labels,0.1,epochs[index]+1,None,None)
print("")
print("LAST EPOCH")
print(ep, 'epoch, CE = {:6.2%}, RE = {:.5f}'.format(train_CE, train_RE))
print("------------------------------------------")

#test data
CE, RE, predicted_labels = model.test(testDataValues,testDataLabels)
print("")
print("Test Data")
print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))

#graphs
plt.scatter(testDataValues[:,0],testDataValues[:,1],c = encode_color(testDataLabels))
plt.show()

plt.scatter(testDataValues[:,0],testDataValues[:,1],c = encode_color(predicted_labels))
plt.show()

#error vs time
plt.plot(np.arange(0,ep+1,1),train_CEs,np.arange(0,ep+1,1),train_REs)
plt.title('CE, RE')
plt.legend(['CE', 'RE'])
plt.xlabel('Epoch')
plt.show()

#confusion matrix
cell_text = np.zeros((3,3))
encodeTestDataLabels = encode_letter(testDataLabels)
encodePredictedLabels = encode_letter(predicted_labels)

for i in range(len(testDataLabels)):
    cell_text[encodeTestDataLabels[i]][encodePredictedLabels[i]] += 1

print("")
print("Confucion Matrix")
print("     Predicted Class")
print("     |A|\t\t|B|\t\t|C|")
print("| A |",cell_text[0][0],"\t",cell_text[0][1],"\t",cell_text[0][2])
print("| B |",cell_text[1][0],"\t    ",cell_text[1][1],"\t",cell_text[1][2], "     Actual Class")
print("| C |",cell_text[2][0],"\t    ",cell_text[2][1],"\t",cell_text[2][2])
