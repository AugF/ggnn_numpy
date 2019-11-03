import numpy as np


class CrossEntropyLoss:
    """cross entropy loss"""
    def __init__(self, label):
        self.label = label

    def forward(self, y_pred):
        self.y_pred = y_pred
        loss = - self.y_pred[self.label - 1] + np.log(np.sum(self.y_pred))
        return loss

    def backward(self):  # gradient
        # onehot
        y_onehot = np.zeros((len(self.y_pred),))
        y_onehot[self.label - 1] = 1
        return - np.multiply(y_onehot, self.y_pred) + self.softmax(self.y_pred)

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)


def numerical_grad_2d(f, X, h=1e-3):
    """under the very small number, h should be large"""
    grad = np.zeros(X.shape)
    m, n = X.shape
    for i in range(m):
        for j in range(n):
            X[i, j] += h
            loss1 = f(X)
            X[i, j] -= (2.0*h)
            loss2 = f(X)
            grad[i, j] = (loss1 - loss2) / (2.0*h)
            X[i, j] += h
    return grad


def numerical_grad_1d(f, x, h=1e-5):
    grad = np.zeros((len(x), ))
    for i in range(len(x)):
        x[i] += h
        loss1 = f(x)
        x[i] -= 2 * h
        loss2 = f(x)
        grad[i] = (loss1 - loss2) / (2 * h)
        x[i] += h
    return grad


def test_numerical_1d():
    input = np.array([3., 2.])
    print(numerical_grad_1d(lambda x: np.sum(x ** 2), input))


def test_forward():
    np.random.seed(1)
    y_pred = np.random.random((4, ))
    label = np.arange(5)
    print("y_pred", y_pred)
    print("label", label)
    C = CrossEntropyLoss(label)
    print(C.forward(y_pred))

    import torch.nn as nn
    import torch
    criterion = nn.CrossEntropyLoss()
    loss = criterion(torch.from_numpy(y_pred), torch.from_numpy(label))
    print(loss)


def test_backward():
    np.random.seed(1)
    y_pred = np.random.random((4,))
    label = 2
    print("y_pred", y_pred)
    C = CrossEntropyLoss(label)
    f = lambda x: C.forward(x)

    n_grad = numerical_grad_1d(f, y_pred, h=1e-3)
    grad = C.backward()

    print("n_grad", n_grad)
    print("grad", grad)


if __name__ == '__main__':
    test_forward()