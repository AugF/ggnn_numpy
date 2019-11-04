import numpy as np
import torch.nn as nn
import torch
from utils.tools import onehot, softmax, wrapper


class LossLayer:
    """cross entropy loss"""
    def __init__(self, outputs, y_onehot):
        self.outputs = outputs
        self.y_onehot = y_onehot
        self.softmax_x = softmax(outputs)
        self.n = outputs.shape[0]

    def forward(self):
        cross_sum = - np.multiply(self.y_onehot, np.log(self.softmax_x))
        cross_sum = wrapper(np.sum(cross_sum, axis=1)) # change column to row
        return np.mean(cross_sum)

    def backward(self):  # gradient
        grad = self.softmax_x - self.y_onehot
        return grad / self.n


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
    y_pred = np.random.random((1, 4))
    y = np.array([1]).reshape(1, )
    print("y_pred", y_pred, y_pred.shape)
    print("y", y, y.shape)

    output = torch.from_numpy(y_pred)
    label = torch.from_numpy(y).long()
    # torch loss
    criterion = nn.LossLayer()
    torch_loss = criterion(output, label)
    print("", torch_loss)

    # my loss
    score = output[0, label.item()].item()  # label对应的class的logits（得分）
    print('Score for the ground truth class = ', label)
    first = - score
    second = 0
    for i in range(4):
        second += np.exp(output[0, i])
    second = np.log(second)
    my_loss = first + second
    print('-' * 20) # study
    print('my loss = ', my_loss)

    # numpy loss
    cross_entropy = LossLayer(y_pred, onehot(y,  y_pred.shape[1])) # shape[1]
    print("numpy loss", cross_entropy.forward())


# def test_backward():
if __name__ == '__main__':
    np.random.seed(1)
    y_pred = np.random.random((1, 4))  # reshape(1, -1) 或  reshape(-1, 1)
    y = np.array([1]).reshape(1, )
    y_onehot = onehot(y, y_pred.shape[1])
    print("y_pred", y_pred, y_pred.shape)
    print("y", y, y.shape)

    f = lambda x: LossLayer(x, y_onehot).forward()
    print("numerical gradient", numerical_grad_2d(f, y_pred, h=1e-5))

    print("numpy gradient", LossLayer(y_pred, y_onehot).backward())


# if __name__ == '__main__':
#     test_backward()