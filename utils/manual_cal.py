from utils.loss import LossLayer
from utils.tools import sigmoid, onehot
state_dim = 3
annotation_dim = 1
n_node = 4
n_edge_types = 2

import numpy as np

np.random.seed(123)
annotation = np.random.random((n_node, annotation_dim))
pre_state = np.hstack(annotation, np.zeros((n_node, state_dim - annotation_dim)))
adj = np.random.random((n_node, n_node * n_edge_types * 2))
y = np.array([1]).reshape(1, )

# A. forward
# 1. add weight
weight_in = np.zeros((n_edge_types, state_dim, state_dim))
weight_out = np.zeros((n_edge_types, state_dim, state_dim))
in_states = np.zeros((n_edge_types, n_node, state_dim))
out_states = np.zeros((n_edge_types, n_node, state_dim))

for i in range(n_edge_types):
    in_states[i] = np.matmul(pre_state, weight_in[i])
    out_states[i] = np.matmul(pre_state, weight_out[i])

# global
a_in_t = np.zeros((n_node, state_dim))
a_out_t = np.zeros((n_node, state_dim))
for i in range(n_edge_types):
    a_in_t += np.matmul(adj[:, i * n_node: (i + 1) * n_node], in_states[i])
    a_out_t += np.matmul(adj[:, (i + n_edge_types) * n_node: (i + 1 + n_edge_types) * n_node], out_states[i])


# ggsnn
# 1. propogator model
weight_z = np.zeros((3, state_dim, state_dim))
weight_r = np.zeros((3, state_dim, state_dim))
weight_h = np.zeros((3, state_dim, state_dim))

z_t = sigmoid(np.matmul(a_in_t, weight_z[0]) + np.matmul(a_out_t, weight_z[1]) + np.matmul(pre_state, weight_z[2]))
r_t = sigmoid(np.matmul(a_in_t, weight_r[0]) + np.matmul(a_out_t, weight_r[1]) + np.matmul(pre_state, weight_r[2]))
h_zt = np.tanh(np.matmul(a_in_t, weight_h[0]) + np.matmul(a_out_t, weight_h[1]) + np.matmul(pre_state, weight_h[2]))
h_t = (1 - z_t) * pre_state + z_t * h_zt

# 2. output model
weight_ho = np.zeros((state_dim, state_dim))
weight_xo = np.zeros((annotation_dim, state_dim))
weight_o = np.zeros((state_dim, 1))

z_1 = np.tanh(np.matmul(h_t, weight_ho) + np.matmul(annotation, weight_xo))
z = np.matmul(z_1, weight_o)

# loss
loss_fun = LossLayer(z.reshape(1, -1), onehot(y, n_node))
loss = loss_fun.forward()

# B. backward
# 1. grad loss
grad_z = loss_fun.backward().reshape(-1, 1)

# 2. grad output
grad_weight_o = np.matmul(z_1.T, grad_z)
t = np.matmul(grad_z, weight_o.T) * (1 - z_1 ** 2)
grad_weight_xo = np.matmul(annotation.T, t)
grad_weight_ho = np.matmul(h_t.T, t)

grad_ht = np.matmul(t, weight_ho.T)

# 2. grad propogation
grad_1 = grad_ht * z_1 * (1 - h_t ** 2)
# grad weight h
grad_weight_h_x = np.matmul(pre_state.T, grad_1)
grad_weight_h_in = np.matmul(a_in_t.T, grad_1)
grad_weight_h_out = np.matmul(a_out_t.T, grad_1)

# grad weight r
grad_2 = np.matmul(grad_1, weight_h[2].T) * pre_state * r_t * (1 - r_t)
# grad weight r
grad_weight_r_x = np.matmul(pre_state.T, grad_2)
grad_weight_r_in = np.matmul(a_in_t.T, grad_2)
grad_weight_r_out = np.matmul(a_out_t.T, grad_2)

# grad weight z
grad_3 = grad_ht * (-pre_state + h_zt) * z_t * (1 - z_t)
grad_weight_z_x = np.matmul(pre_state.T, grad_3)
grad_weight_z_in = np.matmul(a_in_t.T, grad_3)
grad_weight_z_out = np.matmul(a_out_t.T, grad_3)

# a_in_t, a_out_t
grad_a_in_t = np.matmul(grad_1, weight_h[0].T) + np.matmul(grad_2, weight_r[0]) + \
              np.matmul(grad_3, weight_z[0].T)

grad_a_out_t = np.matmul(grad_1, weight_h[1].T) + np.matmul(grad_2, weight_r[1].T) + np.matmul(grad_3, weight_z[1].T)

# grad per state: part1
grad_pre_state = np.matmul(grad_1, weight_h[2].T) + np.matmul(grad_2, weight_r[2].T) + np.matmul(grad_ht, weight_z[2].T) \
                 + grad_ht * (1 - z_t)

grad_weight_in = np.zeros((n_edge_types, state_dim, state_dim))
grad_weight_out = np.zeros((n_edge_types, state_dim, state_dim))

# grad per state: part2
for i in range(n_edge_types):
    t1 = np.matmul(adj[:, i * n_node: (i + 1) * n_node].T, grad_a_in_t)
    t2 = np.matmul(adj[:, (i + n_edge_types) * n_node: (i + 1 + n_edge_types) * n_node].T, grad_a_out_t)
    grad_weight_in[i] = np.matmul(pre_state.T, t1)
    grad_weight_out[i] = np.matmul(pre_state.T, t2)
    grad_pre_state = grad_pre_state + np.matmul(t1, in_states[i].T) + np.matmul(t2, out_states[i].T)
