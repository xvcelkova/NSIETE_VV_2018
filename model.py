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
                h = list()  #vstupy pred maticovym nasobenim
                b = list()  #vysledok vstupov * matica vah, pred aktiv fciou
                gradients = [0] * len(layers)
                dW = [0] * len(layers)
                y = values[i]
                d = targets[...,i]

                for j in range(len(layers)):
                    h.append(y)
                    y = np.append(y, [1], 0)
                    a = np.matmul(self.weights[j], y)
                    b.append(a)
                    y = self.f_sig(a);

                CE += labels[i] != onehot_decode(y)
                RE += self.cost(d, y)
                gradients[2] = 3

                #backpropagation
                for j in reversed(range(len(layers))):
                    if(j==(len(layers)-1)):
                        gradients[j] = ((d - y) * self.df_sig(b[j]))
                    else:
                        gradients[j] = np.matmul(self.df_sig(b[j]), gradients[j+1])

                for j in reversed(range(len(layers))):
                    if(j==len(layers)-1):
                        dW[j] = np.outer(gradients[j], h[j])
                        self.weights[j] += alpha * dW[j]
                    else:
                        dW[j] = np.outer(gradients[j], b[j])
                        self.weights[j] += alpha * dW[j]

            CE /= dim
            RE /= dim
            print('CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))