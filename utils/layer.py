import numpy as np
from utils.tools import sigmoid, onehot, softmax, wrapper
from utils.optimizer import Adam

class LossLayer:
    def __init__(self, n_node):
        self.n_node = n_node

    def forward(self, z, y):
        """
        :param z:  (n_node, 1)
        :param y: 1
        :return: loss
        """
        z = z.reshape(1, -1)
        self.y_onehot = onehot(y, self.n_node)
        self.n = z.shape[0]
        self.softmax_z = softmax(z)
        cross_sum = - np.multiply(self.y_onehot, np.log(self.softmax_z))
        cross_sum = wrapper(np.sum(cross_sum, axis=1)) # change column to row
        return np.mean(cross_sum)

    def backward(self):
        grad = self.softmax_z - self.y_onehot
        grad = grad / self.n
        return grad.reshape(-1, 1)


class OutLayer:
    def __init__(self, annotation_dim, state_dim, lr=0.005):
        self.annotation_dim = annotation_dim
        self.state_dim = state_dim
        self.weight_ho = np.random.random((state_dim, state_dim))
        self.weight_xo = np.random.random((annotation_dim, state_dim))
        self.weight_o = np.random.random((state_dim, 1))

        self.adam_weight_ho = Adam(weights=self.weight_ho, learning_rate=lr)
        self.adam_weight_xo = Adam(weights=self.weight_xo, learning_rate=lr)
        self.adam_weight_o = Adam(weights=self.weight_o, learning_rate=lr)

    def forward(self, ht, annotation):
        """
        :param ht: (n_node, state_dim)
        :param annotation: (n_node, annotation_dim)
        :return: (n_node, 1)
        """
        self.ht, self.annotation = ht, annotation
        self.z_1 = np.tanh(np.matmul(ht, self.weight_ho) + np.matmul(annotation, self.weight_xo))
        self.z = np.matmul(self.z_1, self.weight_o)
        return self.z

    def backward(self, grad_z):
        t1 = np.matmul(grad_z, self.weight_o.T) * (1 - self.z_1 ** 2)
        grad_weight_o = np.matmul(self.z_1.T, grad_z)
        grad_weight_xo = np.matmul(self.annotation.T, t1)
        grad_weight_ho = np.matmul(self.ht.T, t1)
        grad_ht = np.matmul(t1, self.weight_ho.T)
        # update
        self.weight_ho = self.adam_weight_ho.minimize(grad_weight_ho)
        self.weight_xo = self.adam_weight_xo.minimize(grad_weight_xo)
        self.weight_o = self.adam_weight_o.minimize(grad_weight_o)
        return grad_ht


class PropogatorLayer:
    """maybe loop, so backward and grad should apart"""
    def __init__(self, state_dim, lr=0.005):
        self.lr = lr
        self.state_dim = state_dim
        self.weight_z = np.random.random((3, state_dim, state_dim))
        self.weight_r = np.random.random((3, state_dim, state_dim))
        self.weight_h = np.random.random((3, state_dim, state_dim))

        self.grad_weight_z = np.zeros(self.weight_z.shape)
        self.grad_weight_r = np.zeros(self.weight_r.shape)
        self.grad_weight_h = np.zeros(self.weight_h.shape)

        self.adam_weight_z = Adam(weights=self.grad_weight_z, learning_rate=lr)
        self.adam_weight_r = Adam(weights=self.grad_weight_r, learning_rate=lr)
        self.adam_weight_h = Adam(weights=self.grad_weight_h, learning_rate=lr)

    def forward(self, pre_state, a_in_t, a_out_t):
        """
        :param pre_state: (n_node, state_dim)
        :param a_in_t: (n_node, state_dim)
        :param a_out_t: (n_node, state_dim)
        :return: ht (n_node, state_dim)
        """
        self.pre_state = pre_state
        self.a_in_t = a_in_t
        self.a_out_t = a_out_t
        self.z_t = sigmoid(np.matmul(a_in_t, self.weight_z[0]) +
                           np.matmul(a_out_t, self.weight_z[1]) +
                           np.matmul(pre_state, self.weight_z[2]))
        self.r_t = sigmoid(np.matmul(a_in_t, self.weight_r[0]) +
                           np.matmul(a_out_t, self.weight_r[1]) +
                           np.matmul(pre_state, self.weight_r[2]))
        self.h_zt = np.tanh(np.matmul(a_in_t, self.weight_h[0]) +
                            np.matmul(a_out_t, self.weight_h[1]) +
                            np.matmul(pre_state * self.r_t, self.weight_h[2]))
        self.h_t = (1 - self.z_t) * pre_state + self.z_t * self.h_zt
        return self.h_t

    def backward(self, grad_ht):
        grad_h = grad_ht * self.z_t * (1 - self.h_zt ** 2)
        grad_r = np.matmul(grad_h, self.weight_h[2].T) * self.pre_state * \
                 self.r_t * (1 - self.r_t)
        grad_z = grad_ht * (-self.pre_state + self.h_zt) * self.z_t * (1 - self.z_t)

        grad_weight_z = np.zeros(self.weight_z.shape)
        grad_weight_r = np.zeros(self.weight_r.shape)
        grad_weight_h = np.zeros(self.weight_h.shape)
        for i, arr in enumerate([self.a_in_t, self.a_out_t, self.pre_state]):
            grad_weight_h[i] = np.matmul(arr.T, grad_h)
            grad_weight_r[i] = np.matmul(arr.T, grad_r)
            grad_weight_z[i] = np.matmul(arr.T, grad_z)

        grad_a_in_t = np.zeros(self.a_in_t.shape)
        grad_a_out_t = np.zeros(self.a_out_t.shape)
        grad_pre_state = grad_ht * (1 - self.z_t)

        for i, arr in enumerate([grad_a_in_t, grad_a_out_t, grad_pre_state]):
            arr += np.matmul(grad_r, self.weight_r[i].T) + \
                    np.matmul(grad_z, self.weight_z[i].T)
            t = np.matmul(grad_h, self.weight_h[i].T)
            if i == 2:
                arr += t * self.r_t
            else:
                arr += t
        return grad_a_in_t, grad_a_out_t, grad_pre_state, grad_weight_z, grad_weight_r, grad_weight_h

    def update(self):
        self.weight_z = self.adam_weight_z.minimize(self.grad_weight_z)
        self.weight_r = self.adam_weight_r.minimize(self.grad_weight_r)
        self.weight_h = self.adam_weight_h.minimize(self.grad_weight_h)


class GlobalLayer:
    def __init__(self, n_edge_types, n_node, state_dim, lr=0.005):
        self.n_edge_types = n_edge_types
        self.state_dim = state_dim
        self.n_node = n_node
        self.lr = lr
        self.weight_in = np.random.random((n_edge_types, state_dim, state_dim))
        self.weight_out = np.random.random((n_edge_types, state_dim, state_dim))

        self.grad_weight_in = np.zeros(self.weight_in.shape)
        self.grad_weight_out = np.zeros(self.weight_out.shape)
        
        self.adam_weight_in = Adam(weights=self.weight_in, learning_rate=lr)
        self.adam_weight_out = Adam(weights=self.weight_out, learning_rate=lr)

    def forward(self, pre_state, adj):
        """
        :param pre_state: (n_node, state_dim)
        :param adj: (n_node, n_node * n_edge_types * 2)
        :return: a_in_t, a_out_t: (n_node, state_dim)
        """
        self.pre_state, self.adj = pre_state, adj
        self.in_states = np.zeros((self.n_edge_types, self.n_node, self.state_dim))
        self.out_states = np.zeros((self.n_edge_types, self.n_node, self.state_dim))

        for i in range(self.n_edge_types):
            self.in_states[i] = np.matmul(pre_state, self.weight_in[i])
            self.out_states[i] = np.matmul(pre_state, self.weight_out[i])

        a_in_t = np.zeros((self.n_node, self.state_dim))
        a_out_t = np.zeros((self.n_node, self.state_dim))
        for i in range(self.n_edge_types):
            a_in_t += np.matmul(adj[:, i * self.n_node: (i + 1) * self.n_node], self.in_states[i])
            a_out_t += np.matmul(adj[:, (i + self.n_edge_types) * self.n_node: (i + 1 + self.n_edge_types) * self.n_node], self.out_states[i])
        return a_in_t, a_out_t

    def backward(self, grad_a_in_t, grad_a_out_t, grad_pre_state):
        grad_weight_in = np.zeros(self.weight_in.shape)
        grad_weight_out = np.zeros(self.weight_out.shape)

        for i in range(self.n_edge_types):
            t1 = np.matmul(self.adj[:, i * self.n_node: (i + 1) * self.n_node].T, grad_a_in_t)
            t2 = np.matmul(self.adj[:, (i + self.n_edge_types) * self.n_node: (i + 1 + self.n_edge_types) * self.n_node].T, grad_a_out_t)
            grad_weight_in[i] = np.matmul(self.pre_state.T, t1)
            grad_weight_out[i] = np.matmul(self.pre_state.T, t2)
            grad_pre_state += np.matmul(t1, self.weight_in[i].T) + np.matmul(t2, self.weight_out[i].T)

        return grad_pre_state, grad_weight_in, grad_weight_out

    def update(self):
        self.weight_in = self.adam_weight_in.minimize(self.grad_weight_in)
        self.grad_weight_out = self.adam_weight_out.minimize(self.grad_weight_out)