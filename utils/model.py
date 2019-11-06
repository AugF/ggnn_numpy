import numpy as np
from utils.layer import *

class GGNN:
    """1. propagation step  2. output step; without grad update"""
    def __init__(self, annotation_dim, state_dim, n_node, n_edge_types, n_steps=5, lr=0.01):
        self.annotation_dim = annotation_dim
        self.state_dim = state_dim
        self.n_node = n_node
        self.n_edge_types = n_edge_types
        self.n_steps = n_steps  # 隐藏层的个数
        self.lr = lr

        self.GlobalLayer = GlobalLayer(n_edge_types, n_node, state_dim, lr)
        self.PropogatorLayer = PropogatorLayer(state_dim, lr)
        self.OutLayer = OutLayer(annotation_dim, state_dim, lr)
        self.LossLayer = LossLayer(n_node)

    def forward(self, pre_state, annotation, adj, mode, target=None):
        # pre_state = np.hstack((annotation, np.zeros((self.n_node, self.state_dim - self.annotation_dim))))
        # pre_state = annotation
        for i in range(1):
            a_in_t, a_out_t = self.GlobalLayer.forward(pre_state, adj)
            pre_state = self.PropogatorLayer.forward(pre_state, a_in_t, a_out_t)

        self.z = self.OutLayer.forward(pre_state, annotation)
        loss = 0
        if mode == "train":
            loss = self.LossLayer.forward(self.z, target)
        return loss

    def backward(self):
        """share the GRU part, so should check"""
        grad_z = self.LossLayer.backward()
        grad_ht = self.OutLayer.backward(grad_z)
        for i in range(1):
            grad_res = self.PropogatorLayer.backward(grad_ht)
            self.PropogatorLayer.grad_weight_z += grad_res[3]
            self.PropogatorLayer.grad_weight_r += grad_res[4]
            self.PropogatorLayer.grad_weight_h += grad_res[5]
            grad_global_res = self.GlobalLayer.backward(grad_res[0], grad_res[1], grad_res[2])
            self.GlobalLayer.grad_weight_in += grad_global_res[1]
            self.GlobalLayer.grad_weight_out += grad_global_res[2]
            grad_ht = grad_global_res[0]
        self.PropogatorLayer.update()
        self.GlobalLayer.update()
        return grad_ht
