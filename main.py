from utils.dataset import bAbIDataset
from utils.model import GGNN
from utils.loss import CrossEntropyLoss
import numpy as np

task_id = 4
processed_path = 'processed'
batch_size = 1
train_size = 50
qustion_id = 0
workers = 2
state_dim = 4  # p
annotation_dim = 1  # p
n_steps = 5
lr = 0.01

dataroot = 'babi_data/{}/train/{}_graphs.txt'.format(processed_path, task_id)

# 1. train
train_dataset = bAbIDataset(dataroot, qustion_id, True, train_size)
# data: (am, annotation, target);  n_edge_types, n_node
n_edge_types = train_dataset.n_edge_types
n_node = train_dataset.n_node

net = GGNN()
criterion = CrossEntropyLoss

for i, (adj_matrix, annotation, target) in enumerate(train_dataset.data):
    # x: (4, 1) padding: (4, 3)
    init_inputs = np.stack((annotation, np.zeros(n_node, state_dim - annotation_dim))) # init_inputs
    outputs = net
    loss_fun = criterion(outputs, target)
    loss = loss_fun.forward()
    loss_grad = loss_fun.backward()

# 2. test
test_dataset = bAbIDataset(dataroot, qustion_id, False, train_size)

correct = 0
loss = 0
count = 0

for i, (adj_matrix, annotation, target) in enumerate(test_dataset.data):
    outputs = net  # (1, 4)
    loss += criterion(outputs, target).forward()
    correct += int(np.argmax(outputs.reshape(-1, )) == target)
    count += 1

acc = correct * 1.0 / count
avg_loss = loss / count

print("test_acc: ", acc, "avg_acc: ", avg_loss)