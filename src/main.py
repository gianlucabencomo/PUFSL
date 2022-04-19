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
from initialize import *
from model import ProtoNet
from loss import proto_loss
from sampler import BatchSampler


def train(config, train, val, model, optim, lr_scheduler):
    '''
    Training Loop.
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    if len(val) == 0:
        best_state = None
    best_acc = 0

    best_model_path = os.path.join(config.experiment_root, 'best_model_' + 
                                                str(config.seed) + '.pth')
    last_model_path = os.path.join(config.experiment_root, 'last_model_' + 
                                                str(config.seed) + '.pth')
   
    for epoch in range(config.epochs): 
        print('=== Epoch: {} ==='.format(epoch))
        
        pos = config.pos_support + config.pos_query
        neg = config.neg_support + config.neg_query
        data = BatchSampler(train, pos, neg, config.iterations)
        
        model.train()
        for batch in data:    
            optim.zero_grad()
            model_output = []
            y = []
            for graph in batch:
                model_output.append(model(graph))
                y.append(graph.y)
            model_output = torch.cat(model_output, dim=0)
            y = torch.Tensor(y)
            loss, acc = proto_loss(model_output, target=y,
                                    pos_support=config.pos_support, 
                                    neg_support=config.neg_support)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss)
        avg_acc = np.mean(train_acc)
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
    
        if len(val) == 0:
            continue
        
        data = BatchSampler(val, pos, neg, config.iterations)
        model.eval()
        for batch in data:
            model_output = []
            y = []
            for graph in batch:
                model_output.append(model(graph))
                y.append(graph.y)
            model_output = torch.cat(model_output, dim=0)
            y = torch.Tensor(y)
            loss, acc = proto_loss(model_output, target=y,
                                    pos_support=config.pos_support,
                                    neg_support=config.neg_support)
            val_loss.append(loss.item())
            val_acc.append(acc.item()) 
        avg_loss = np.mean(val_loss)
        avg_acc = np.mean(val_acc)
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, 
                                                    avg_acc, postfix)) 
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
         
    last_state = model.state_dict()
    torch.save(last_state, last_model_path)
    
    return last_state, best_state, best_acc, train_loss, train_acc, val_loss, val_acc
        

# collect options
config = get_config().parse_args()
if not os.path.exists(config.experiment_root):
    os.makedirs(config.experiment_root)
# TODO: check save/data paths and mkdir or whatever if not there


# set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'\nDevice for train/test : {device}')

# initialize seed
init_seed(config.seed)
print(f'Random seed : {config.seed}')

# load dataset
print(f'\nLoading Training, Validation, and Testing Data... ({config.dataset})')
train_data, val_data, test_data = init_data(config)
print('Data Loading complete.\n')
print('Training Set')
print('=========================')
print(f'Number of Tasks: {len(train_data)}\n')
print('Validation Set')
print('=========================')
print(f'Number of Tasks: {len(val_data)}\n')
print('Test Set')
print('=========================')
print(f'Number of Tasks: {len(test_data)}\n')


# TODO: initilize protonet model, optimizer, learning rate scheduler
model = ProtoNet(train_data[0].num_features, config.out_channels).to(device)
optim = torch.optim.Adam(params=model.parameters(),
                            lr=config.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=config.lr_scheduler_gamma,
                                           step_size=config.lr_scheduler_step)

# TODO: call episodic training loop
print('Starting training loop...\n')
res = train(config=config,
                train=train_data,
                val=val_data,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
print('\nTraining Complete!\n')

# HEY I WAS HERE WHEN I TOOK A BREAK. 
print('Testing with last model..')
test(config=config,
        test_dataloader=test_dataloader,
        model=model)

model.load_state_dict(best_state)
print('Testing with best model..')
test(config=config,
        test_dataloader=test_dataloader,
        model=model)

# TODO: test model (write a function for test, same for train)

# TODO: test last model + best model (save best model)

# 
