import os
import time
import torch
import numpy as np

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data.collate import collate

from data import *
from config import get_config
from helper import *
from model import ProtoNet
from loss import proto_loss

def sampler(tasks, pos, neg):
    D = []
    for task in tasks:
        samp = np.array(range(len(task)))
        np.random.shuffle(samp)
         
        i = 0
        while (i < len(samp)):
            pos_count = 0
            neg_count = 0
            data = []
            
            while (pos_count + neg_count < pos + neg):
                y = task[samp[i]].y.numpy().item()
                if (pos_count < pos and y == 1):
                    data.append(task[samp[i]])
                    pos_count += 1
                elif (neg_count < neg and y == 0):
                    data.append(task[samp[i]])
                    neg_count += 1
                i += 1
                if (i == len(samp)):
                    break
            if (i == len(samp)):
                break
 
            if len(data) == 1:
                    return data[0]
           
            np.random.shuffle(data) 
            data, _, _ = collate(
                    data[0].__class__,
                    data_list=data,
                    increment=False,
                    add_batch=False,
                )

            D.append(data)

    np.random.shuffle(D)
    return D
            
     

def train(config, train, model, optim, lr_scheduler):
    '''
    Training Loop.
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = -1

    best_model_path = os.path.join(config.experiment_root, 'best_model.pth')
   
    for epoch in range(config.epochs): 
        print('=== Epoch: {} ==='.format(epoch))
        
        data = sampler(train, 10, 10)

        model.train()
        for batch in data:    
            optim.zero_grad()
            model_output = model(batch)
            loss, acc = proto_loss(model_output, target=batch.y,
                                    n_support=5)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            #print(train_loss) 
            train_acc.append(acc.item())
            #print(train_acc)



lengths =   {
            'tox21': 12,
            'pcba': 128,
            'toxcast_dataset': 618,
            'muv': 17,
            }

split = 0.8


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

train_data = []
test_data = []
for i in range(0, lengths[config.dataset]):

    if i < int(lengths[config.dataset] * split):
        train_data.append(MoleculeNet(root=config.data_path, name=config.dataset,
                                task=i, 
                                transform=transform, 
                                pre_transform=pre_transform, 
                                pre_filter=pre_filter))
    else:        
        test_data.append(MoleculeNet(root=config.data_path, name=config.dataset,
                                task=i,
                                transform=transform, 
                                pre_transform=pre_transform, 
                                pre_filter=pre_filter))

print(len(test_data))
print(len(train_data))

print(train_data[1].data)
print(train_data[2].data)
print(train_data[3].data)
print(train_data[4].data)
print(train_data[5].data.y)
print(train_data[6].data.y)

#print(train_data.data)
#print(train_data[[1,2,3]])
inds = []
count = 0
task = 0

print(np.isnan(train_data[0][2].y.numpy().item()))


# TODO: initilize protonet model, optimizer, learning rate scheduler
model = ProtoNet(train_data[0].num_features, 64).to(device)
optim = torch.optim.Adam(params=model.parameters(),
                            lr=config.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=config.lr_scheduler_gamma,
                                           step_size=config.lr_scheduler_step)

# TODO: call episodic training loop
train(config=config,
                train=train_data,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)

# TODO: test model (write a function for test, same for train)

# TODO: test last model + best model (save best model)

# 
