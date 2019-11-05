from utils.tools import numerical_grad_2d
from utils.layer import *


def test_losslayer():
    n_node = 4
    np.random.seed(1)
    z = np.random.random((1, n_node))
    y = np.array([1])
    lossLayer = LossLayer(n_node)

    loss = lossLayer.forward(z, y)
    print("loss", loss)

    f = lambda x: lossLayer.forward(x, y)
    manual_grad = numerical_grad_2d(f, z)
    grad = lossLayer.backward()
    print("manual_grad", manual_grad)
    print("grad", grad)


def test_outLayer():
    np.random.seed(3)
    n_node = 4
    annotation_dim = 1
    state_dim = 3

    annotation = np.random.random((n_node, annotation_dim))
    ht = np.random.random((n_node, state_dim))
    y = np.array([1])
    lossLayer = LossLayer(n_node)
    outLayer = OutLayer(annotation_dim, state_dim)

    z = outLayer.forward(ht, annotation)
    loss = lossLayer.forward(z, y)
    print("loss", loss)

    f = lambda x: lossLayer.forward(outLayer.forward(x, annotation), y)
    manual_grad = numerical_grad_2d(f, ht)
    grad_z = lossLayer.backward()
    grad = outLayer.backward(grad_z)
    print("manual_grad", manual_grad)
    print("grad", grad)


def test_ProgatorLayer():
    np.random.seed(5)
    n_node = 4
    annotation_dim = 1
    state_dim = 3

    annotation = np.random.random((n_node, annotation_dim))
    pre_state = np.random.random((n_node, state_dim))
    a_in_t = np.random.random((n_node, state_dim))
    a_out_t = np.random.random((n_node, state_dim))
    y = np.array([1])
    lossLayer = LossLayer(n_node)
    outLayer = OutLayer(annotation_dim, state_dim)
    propogatorLayer = PropogatorLayer(state_dim)

    ht = propogatorLayer.forward(pre_state, a_in_t, a_out_t)
    z = outLayer.forward(ht, annotation)
    loss = lossLayer.forward(z, y)
    print("loss", loss)

    f_x = lambda x: lossLayer.forward(outLayer.forward(
        propogatorLayer.forward(x, a_in_t, a_out_t), annotation), y)

    f_in = lambda x: lossLayer.forward(outLayer.forward(
        propogatorLayer.forward(pre_state, x, a_out_t), annotation), y)

    f_out = lambda x: lossLayer.forward(outLayer.forward(
        propogatorLayer.forward(pre_state, a_in_t, x), annotation), y)

    manual_grad_x = numerical_grad_2d(f_x, pre_state)
    manual_grad_in = numerical_grad_2d(f_in, a_in_t)
    manual_grad_out = numerical_grad_2d(f_out, a_out_t)

    grad_z = lossLayer.backward()
    grad_ht = outLayer.backward(grad_z)
    grad_a_in_t, grad_a_out_t, grad_pre_state = propogatorLayer.backward(grad_ht)[:3]
    print("manual_grad: \n x: {} \n in: {} \n out: {}".format(manual_grad_x, manual_grad_in, manual_grad_out))
    print("grad: \n x: {} \n in: {} \n out: {}".format(grad_pre_state, grad_a_in_t, grad_a_out_t))


def test_GlobalLayer():
    np.random.seed(8)
    n_node = 4
    annotation_dim = 1
    state_dim = 3
    n_edge_types = 2

    annotation = np.random.random((n_node, annotation_dim))
    pre_state = np.random.random((n_node, state_dim))
    adj = np.random.random((n_node, n_node * n_edge_types * 2))
    y = np.array([1])

    lossLayer = LossLayer(n_node)
    outLayer = OutLayer(annotation_dim, state_dim)
    propogatorLayer = PropogatorLayer(state_dim)
    globalLayer = GlobalLayer(n_edge_types, n_node, state_dim)

    a_in_t, a_out_t = globalLayer.forward(pre_state, adj)
    ht = propogatorLayer.forward(pre_state, a_in_t, a_out_t)
    z = outLayer.forward(ht, annotation)
    loss = lossLayer.forward(z, y)
    print("loss", loss)

    def f(x):
        a_in_t, a_out_t = globalLayer.forward(x, adj)
        return lossLayer.forward(outLayer.forward(
            propogatorLayer.forward(x, a_in_t, a_out_t), annotation), y)

    manual_grad_x = numerical_grad_2d(f, pre_state)


    grad_z = lossLayer.backward()
    grad_ht = outLayer.backward(grad_z)
    grad_a_in_t, grad_a_out_t, grad_pre_state = propogatorLayer.backward(grad_ht)[:3]
    grad_x = globalLayer.backward(grad_a_in_t, grad_a_out_t, grad_pre_state)[0]

    print("manual grad", manual_grad_x)
    print("grad", grad_x)

def test_GlobalLayer2():
    np.random.seed(8)
    n_node = 4
    annotation_dim = 1
    state_dim = 3
    n_edge_types = 2

    annotation = np.random.random((n_node, annotation_dim))
    pre_state = np.random.random((n_node, state_dim))
    adj = np.random.random((n_node, n_node * n_edge_types * 2))

    lossLayer = LossLayer(n_node)
    outLayer = OutLayer(annotation_dim, state_dim)
    propogatorLayer = PropogatorLayer(state_dim)
    globalLayer = GlobalLayer(n_edge_types, n_node, state_dim)

    a_in_t, a_out_t = globalLayer.forward(pre_state, adj)
    loss = np.sum(a_in_t) + np.sum(a_out_t) + np.sum(pre_state)
    print("loss", loss)


    def f(x):
        a_in_t, a_out_t = globalLayer.forward(x, adj)
        return np.sum(a_in_t) + np.sum(a_out_t)


    manual_grad_x = numerical_grad_2d(f, pre_state)

    grad_a_in_t = np.ones(a_in_t.shape)
    grad_a_out_t = np.ones(a_out_t.shape)
    grad_pre_state = np.ones(pre_state.shape)
    grad_x = globalLayer.backward(grad_a_in_t, grad_a_out_t, grad_pre_state)[0]

    print("manual grad", manual_grad_x)
    print("grad", grad_x)

if __name__ == '__main__':
    test_losslayer()
    test_outLayer()
    test_ProgatorLayer()
    test_GlobalLayer()
    test_GlobalLayer2()