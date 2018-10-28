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

def encode_color(labels):
    out = [None] *len(labels)

    for i in range(len(labels)):
        if (labels[i] == "A"):
            out[i] = "r"
        if (labels[i] == "B"):
            out[i] = "g"
        if (labels[i] == "C"):
            out[i] = "b"
    return out

def encode_letter(labels):
    out = [None] *len(labels)

    for i in range(len(labels)):
        if (labels[i] == "A"):
            out[i] = 0
        if (labels[i] == "B"):
            out[i] = 1
        if (labels[i] == "C"):
            out[i] = 2
    return out