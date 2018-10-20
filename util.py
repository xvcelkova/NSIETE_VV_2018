import numpy as np

def onehot_decode(X):
    value = np.argmax(X, axis=0)
    if(value == 0):
        return "A"
    if(value == 1):
        return "B"
    if(value == 2):
        return "C"


def onehot_encode(labels, c):
    n = len(labels)
    out = np.zeros((c, n))

    for i in range(n):
        if(labels[i] == "A"):
            out[0][i] = 1
        if(labels[i] == "B"):
            out[1][i] = 1
        if(labels[i] == "C"):
            out[2][i] = 1
    return out