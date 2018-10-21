import numpy as np

from util import *

class Model():
    def __init__(self,weights):
        self.weights = weights

    #sigmoid
    def f_sig(self, x):
        sigmoid = 1 / (1 + np.e ** (-x))
        return sigmoid

    def df_sig(self, x):
        y = 1 / (1 + np.e ** (-x))
        dsigmoid = y * (1 - y)
        return dsigmoid

    #softmax
    def f_softmax(self, x):
        softmax = np.e ** (x) / np.sum(np.e ** (x))
        return softmax

    def df_softmax(self, x):
        y = np.e ** (x) / np.sum(np.e ** (x))
        dsoftmax = y * (1 - y)

    def cost(self,targets,outputs):
        return np.sum((targets - outputs) ** 2, axis=0)

    def train(self,layers, values, labels, alpha, eps):
        (dim, count) = values.shape
        #TODO zmenit 3ku
        targets = onehot_encode(labels,3)

        for ep in range(eps):
            CE = 0
            RE = 0
            for i in np.random.permutation(dim):
                nets = list()  #vysledok vstupov * matica vah, pred aktiv fciou
                activations = list()
                gradients = [None] * len(layers)
                dW = [None] * len(layers)
                y = values[i]
                d = targets[...,i]

                for j in range(len(layers)):
                    y = np.append(y, [1], 0)
                    a = np.matmul(self.weights[j], y)
                    nets.append(a)
                    y = self.f_sig(a)
                    activations.append(y)

                CE += labels[i] != onehot_decode(y)
                RE += self.cost(d, y)

                #backpropagation
                for j in reversed(range(len(layers))):
                    if(j==(len(layers)-1)):
                        gradients[j] = ((d - y) * self.df_sig(nets[j]))
                        dW[j] = np.outer(gradients[j], np.append(activations[j - 1],[1],0))
                    else:
                        gradients[j] = np.matmul(np.transpose(self.weights[j + 1][:, :-1]), gradients[j + 1]) * self.df_sig(nets[j])
                        if(j==0):
                            dW[j] = np.outer(gradients[j], np.append(values[i], [1], 0))
                        else:
                            dW[j] = np.outer(gradients[j], np.append(activations[j - 1], [1], 0))

                for j in range(len(layers)):
                    self.weights[j] += alpha * dW[j]

            CE /= dim
            RE /= dim
            print('CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))