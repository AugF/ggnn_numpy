import numpy as np
from utils.layer import *

class GGNN:
    """1. propagation step  2. output step; without grad update"""
    def __init__(self, annotation_dim, state_dim, n_node, n_edge_types, n_steps, lr):
        self.annotation_dim = annotation_dim
        self.state_dim = state_dim
        self.n_node = n_node
        self.n_edge_types = n_edge_types
        self.n_steps = n_steps  # 隐藏层的个数
        self.lr = lr

        self.GlobalLayer = GlobalLayer(n_edge_types, n_node, state_dim)
        self.PropogatorLayer = PropogatorLayer(state_dim)
        self.OutLayer = OutLayer(annotation_dim, state_dim)
        self.LossLayer = LossLayer(n_node)

    def forward(self, annotation, adj, target):
        pre_state = np.hstack((annotation, np.zeros((self.n_node, self.state_dim - self.annotation_dim))))

        for i in range(self.n_steps):
            a_in_t, a_out_t = self.GlobalLayer.forward(pre_state, adj)
            pre_state = self.PropogatorLayer.forward(pre_state, a_in_t, a_out_t)

        z = self.OutLayer.forward(pre_state, annotation)
        loss = self.LossLayer.forward(z, target)
        return loss

    def backward(self):
        """share the GRU part, so should check"""
        grad_z = self.LossLayer.backwad()
        grad_ht = self.OutLayer.backward(grad_z)
        for i in range(self.n_steps - 1):
            pass # weight should add together
