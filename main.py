from utils.dataset import bAbIDataset
from utils.model import GGNN
from utils.optimizer import Adam

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

# 1. get data
train_dataset = bAbIDataset(dataroot, qustion_id, True, train_size)
# data: (am, annotation, target);  n_edge_types, n_node
n_edge_types = train_dataset.n_edge_types
n_node = train_dataset.n_node

net = GGNN()


test_dataset = bAbIDataset(dataroot, qustion_id, False, train_size)

# 2. train

# 3. adam loss, weight update

# 4. test
