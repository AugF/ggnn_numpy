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


def numerical_grad_2d(f, X, h=1e-5):
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

def save_weight(file_str, target):
    file_path = "../_weights/{}.txt".format(file_str)
    m, n = target.shape
    with open(file_path, "w") as f:
        for i in range(m):
            for j in range(n):
                f.write(str(target[i, j]) + " ")
            f.write("\n")


def get_Weight_from_file(shape, weight_str):
    m, n = shape
    ans = np.zeros((m, n), dtype=np.float32)
    weight_path = "../_weights/{}.txt".format(weight_str)
    with open(weight_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            for j, v in enumerate(line.strip().split(" ")):
                ans[i, j] = float(v)
    return ans

if __name__ == '__main__':
    x = np.random.random((2, 2))
    f = lambda x: np.sum(np.tanh(x))
    print("1", numerical_grad_2d(f, x))
    print("2", 1 - np.tanh(x) ** 2)