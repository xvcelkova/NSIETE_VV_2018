import numpy as np

from util import *

class Model():
    def __init__(self,weights,layers,functions):
        self.weights = weights
        self.layers = layers
        self.functions = functions

    #sigmoid
    def f_sig(self, x):
        sigmoid = 1 / (1 + np.e ** (-x))
        return sigmoid

    def df_sig(self, x):
        y = self.f_sig(x)
        dsigmoid = y * (1 - y)
        return dsigmoid

    #softmax
    def f_softmax(self, x):
        softmax = np.e ** (x) / np.sum(np.e ** (x))
        return softmax

    def df_softmax(self, x):
        y = self.f_softmax(x)
        dsoftmax = y * (1 - y)
        return dsoftmax

    #tanh
    def f_tanh(self,x):
        tanh = ((np.e ** x - np.e **(-x))/(np.e ** x + np.e **(-x)))
        return tanh

    def df_tanh(self, x):
        y = self.f_tanh(x)
        dtanh = 1 - (y ** 2)
        return dtanh

    def cost(self,targets,outputs):
        return np.sum((targets - outputs) ** 2, axis=0)

    def crossentropy(self,targets,outputs):
        return (np.sum(targets * np.log(outputs)) * -1)

    def forward(self,y,nets,activations):
        for j in range(len(self.layers)):
            y = np.append(y, [1], 0)
            a = np.matmul(self.weights[j], y)
            nets.append(a)
            if(self.functions[j] == "sigmoid"):
                y = self.f_sig(a)
            if (self.functions[j] == "softmax"):
                y = self.f_softmax(a)
            if (self.functions[j] == "tanh"):
                y = self.f_tanh(a)
            activations.append(y)

        return y, nets, activations

    def predict(self, input):
        y,*_ = self.forward(input, list(),list())
        return y

    def test(self, values, labels):
        targets = onehot_encode(labels, 3)
        predicted_labels = list()
        (dim, count) = values.shape
        CE = 0
        RE = 0

        random = np.random.permutation(dim)
        for i in random:
            x = values[i]
            d = targets[..., i]
            y = self.predict(x)
            CE += labels[i] != onehot_decode(y)
            predicted_labels.append(onehot_decode(y))
            if (self.functions[len(self.functions) - 1] == "sigmoid" or self.functions[len(self.functions) - 1] == "tanh"):
                RE += self.cost(d, y)
            if (self.functions[len(self.functions) - 1] == "softmax"):
                RE += self.crossentropy(d, y)

        CE /= dim
        RE /= dim

        return CE, RE, random, predicted_labels


    def train(self, values, labels, alpha, eps):
        (dim, count) = values.shape
        #TODO zmenit 3ku
        targets = onehot_encode(labels,3)

        for ep in range(eps):
            CE = 0
            RE = 0
            for i in np.random.permutation(dim):
                nets = list()  #vysledok vstupov * matica vah, pred aktiv fciou
                activations = list()
                gradients = [None] * len(self.layers)
                dW = [None] * len(self.layers)
                y = values[i]
                d = targets[...,i]

                y,nets,activations = self.forward(y,nets,activations)

                CE += labels[i] != onehot_decode(y)
                if(self.functions[len(self.functions)-1] == "sigmoid" or self.functions[len(self.functions) - 1] == "tanh"):
                    RE += self.cost(d, y)
                if (self.functions[len(self.functions) - 1] == "softmax"):
                    RE += self.crossentropy(d, y)

                #backpropagation
                for j in reversed(range(len(self.layers))):
                    if(j==(len(self.layers)-1)):
                        if(self.functions[j] == "sigmoid"):
                            gradients[j] = ((d - y) * self.df_sig(nets[j]))
                        if (self.functions[j] == "softmax"):
                            gradients[j] = ((d - y) * self.df_softmax(nets[j]))
                        if (self.functions[j] == "tanh"):
                            gradients[j] = ((d - y) * self.df_tanh(nets[j]))
                        dW[j] = np.outer(gradients[j], np.append(activations[j - 1],[1],0))
                    else:
                        if (self.functions[j] == "sigmoid"):
                            gradients[j] = np.matmul(np.transpose(self.weights[j + 1][:, :-1]),
                                                     gradients[j + 1]) * self.df_sig(nets[j])
                        if (self.functions[j] == "softmax"):
                            gradients[j] = np.matmul(np.transpose(self.weights[j + 1][:, :-1]),
                                                     gradients[j + 1]) * self.df_softmax(nets[j])
                        if (self.functions[j] == "tanh"):
                            gradients[j] = np.matmul(np.transpose(self.weights[j + 1][:, :-1]),
                                                     gradients[j + 1]) * self.df_tanh(nets[j])
                        if(j==0):
                            dW[j] = np.outer(gradients[j], np.append(values[i], [1], 0))
                        else:
                            dW[j] = np.outer(gradients[j], np.append(activations[j - 1], [1], 0))

                for j in range(len(self.layers)):
                    self.weights[j] += alpha * dW[j]

            CE /= dim
            RE /= dim
            print(ep+1,'epoch, CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))