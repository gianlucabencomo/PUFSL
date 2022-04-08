import os
import time
import torch
import numpy as np

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader

from data import *
from config import get_config
from helper import *

# collect options
config = get_config().parse_args()

# TODO: check save/data paths and mkdir or whatever if not there


# set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'\nDevice for train/test : {device}')

# initialize seed
init_seed(config.seed)
print(f'Random seed : {config.seed}')

# TODO: add configs for the following.  for now, keep 'none'
transform = None
pre_transform = None
pre_filter = None

# load dataset
print(f'\nLoading Training and Testing Data... ({config.dataset})')
train_data = MoleculeNet(root=config.data_path, name=config.dataset,
                        mode='train', split=0.8, 
                        transform=transform, 
                        pre_transform=pre_transform, 
                        pre_filter=pre_filter)
test_data = MoleculeNet(root=config.data_path, name=config.dataset,
                        mode='test', split=0.8, 
                        transform=transform, 
                        pre_transform=pre_transform, 
                        pre_filter=pre_filter)
print('Data Loaded.')
print()
print('Training Dataset.')
print('====================')
print(f'Number of graphs (compounds): {len(train_data)}')
print(f'Number of features per atom: {train_data.num_features}')
print(f'Number of features per bond: {train_data.num_edge_features}')
print(f'Number of tasks: {train_data.num_classes}') 

print()
print('Test Dataset.')
print('====================')
print(f'Number of graphs (compounds): {len(test_data)}')
print(f'Number of features per atom: {test_data.num_features}')
print(f'Number of features per bond: {test_data.num_edge_features}')
print(f'Number of tasks: {test_data.num_classes}') 

train_data = train_data.shuffle()
test_data = test_data.shuffle()

# TODO: initilize protonet model, optimizer, learning rate scheduler



# TODO: call episodic training loop

# TODO: test model (write a function for test, same for train)

# TODO: test last model + best model (save best model)

# 
