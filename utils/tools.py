import numpy as np


def onehot(y, c):
    res = np.zeros((len(y), c))
    for i, j in enumerate(y):
        res[i, j] = 1
    return res

def wrapper(x):
    return x.reshape(x.shape[0],)


def softmax(x):
    exp_x = np.exp(x)
    sum_x = np.sum(exp_x, axis=1).reshape(-1, 1)
    return exp_x / sum_x


def sigmoid(x):
    return 1 / (1 + np.exp(- x))

def tanh(x):
    exp_x = np.exp(x)
    exp_neg_x = np.exp(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

if __name__ == '__main__':
    y = np.array([1]).reshape(1, )
    print(onehot(y, 4).shape)