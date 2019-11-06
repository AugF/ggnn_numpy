from utils.dataset import bAbIDataset
from utils.model import GGNN

import numpy as np

task_id = 4
processed_path = 'processed'
batch_size = 1  # todo dataloader?, pack batch_size data ?
train_size = 50  # todo understand the meaning, at most 50? yes,  train_size  val_size: 50
test_size = 50
qustion_id = 0  # 注意从0开始编号
state_dim = 4  # p
annotation_dim = 1  # p
n_steps = 5  #  ggnn model part
n_iters = 10  # iter times
lr = 0.01

dataroot = 'babi_data/{}/train/{}_graphs.txt'.format(processed_path, task_id)

# get dataset
train_dataset = bAbIDataset(dataroot, qustion_id, True, train_size)
val_dataset = bAbIDataset(dataroot, qustion_id, False, train_size)
# data: (am, annotation, target);  n_edge_types, n_node
n_edge_types = train_dataset.n_edge_types
n_node = train_dataset.n_node

net = GGNN(annotation_dim=annotation_dim, state_dim=state_dim,
           n_node=n_node, n_edge_types=n_edge_types,
           n_steps=n_steps, lr=lr)

# 1. train model
print("size", len(train_dataset), len(val_dataset))
for epoch in range(n_iters):
    # train_step
    for i, (adj, annotation, target) in enumerate(train_dataset):
        loss = net.forward(annotation=annotation, adj=adj, mode="train", target=np.array([target]))
        net.backward()
        print("after", net.PropogatorLayer.grad_weight_h)
        print("train_data: [{}/{}] [{}/{}] Loss: {}".format(epoch, n_iters, i, len(train_dataset), loss))
        break
    break
    # val step
    # correct = 0
    # loss = 0
    # for i, (adj, annotation, target) in enumerate(val_dataset):
    #     loss += net.forward(annotation=annotation, adj=adj, mode="train", target=np.array([target]))
    #     correct += int(np.argmax(net.z) == target) # todo check what's wrong
    # acc = correct * 1.0 / len(val_dataset)
    # avg_loss = loss / len(val_dataset)
    # print("val_data: [{}/{}], avg_loss {}".format(correct, len(val_dataset), avg_loss))


# 2. test model
# test_dataroot = 'babi_data/{}/test/{}_graphs.txt'.format(processed_path, task_id)
# test_dataset = bAbIDataset(test_dataroot, qustion_id, False, test_size)
#
# correct = 0
# for i, (adj, annotation, target) in enumerate(test_dataset):
#     net.forward(annotation=annotation, adj=adj, mode="test")
#     correct += int(np.argmax(net.z) == target)
# acc = correct * 1.0 / len(test_dataset)
# print("test_data: [{}/{}] {}".format(correct, len(test_dataset), acc))
