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
    Description: Training loop for k-shot learning scenario.
    Inputs:
        config - see config.py
        train - training set
        val - validation set
        model - initialized model to optimize
        optim - initialized optimizer (adam, typically)
        lr_scheduler - configuration for lr changes over time
    Returns:
        State dictionaries for the best state and the last state models. 
        Accuracies and losses for train, val, and best.
    '''
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
        print('=== Training Epoch: {}/{} ==='.format(epoch + 1, config.epochs))
        
        pos = config.pos_support + config.pos_query
        neg = config.neg_support + config.neg_query
        data = BatchSampler(train, pos, neg, config.iterations)
        print('=== Iterations/Epoch: {} ==='.format(len(data)))        
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

def test(config, test, model, n=20):
    '''
    Description: Takes model and runs n episodes of m iter k-shot learning scenarios.
    The accuracy returned is the accuracy of the model after seeing n*m k-shot learning
    setups on n*m uniformly random draws on tasks within the test dataset.
    Inputs:
        config - (argparse)
        test - tasks to test on
        model - model to run 'forward' on
        n - (int) number of episodes
    Returns:
        Average accuracy of n*m uniformly random draws on k-shot learning episodes of a 
        specific task structure
    '''
    avg_acc = []

    for epoch in range(n):
        if epoch % 5 == 0:
            print('=== Testing Epoch: {}/{} ==='.format(epoch + 1, n))
        
        # key difference during testing is that the query size is larger
        # since we would ideally like a more accurate picture of how each   
        # randomly chosen support set performs.  default is 20x larger than
        # query during training
        pos = config.pos_support + config.test_pos_query
        neg = config.neg_support + config.test_neg_query
      
        data = BatchSampler(test, pos, neg, config.iterations)
        
        assert len(data) > 0, 'pos/neg test query set # might be too high! BatchSampler returned 0'
            
        model.eval()
        for batch in data:
            model_output = []
            y = []
            for graph in batch:
                model_output.append(model(graph))        
                y.append(graph.y)
            model_output = torch.cat(model_output, dim=0)
            y = torch.Tensor(y)
            _, acc = proto_loss(model_output, target=y,
                                    pos_support=config.pos_support,
                                    neg_support=config.neg_support)
            avg_acc.append(acc.item())
             
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))
    
    return avg_acc

def main():
    '''
    Initialize and configure everything needed to run training, validation (optional), and testing.
    '''
    # get configuration and setup directories
    config = get_config().parse_args()
    if not os.path.exists(config.experiment_root):
        os.makedirs(config.experiment_root)

    splits = config.tr_split + config.v_split + config.te_split
    assert splits == 1.0, f'Check train/validation/test splits, must sum to 1.0 (currently adds up to {splits:.4f})'

    # set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'\nDevice for train/test : {device}')

    # initialize seed
    init_seed(config.seed)
    print(f'Random seed : {config.seed}')

    # load dataset
    print(f'\nLoading Training, Validation, and Testing Data... ({config.dataset})')
    train_data, val_data, test_data = init_data(config)
   
    # print basic info regarding train/val/test datasets
    print('Data Loading complete.\n')
    print('Training Set')
    print('=========================')
    print(f'Number of Tasks: {len(train_data)}')
    print(f'Mean # of compounds per task: {np.mean([len(i) for i in train_data]):.3f}')
    print(f'Mean # of atoms per compound: {np.mean([i.data.num_nodes/len(i) for i in train_data]):.3f}')
    print(f'Mean # of bonds per compound: {np.mean([i.data.num_edges/len(i) for i in train_data]):.3f}')
    print(f'Number of atom features: {train_data[0].num_features:.1f}')
    print(f'Number of edge features: {train_data[0].num_edge_features:.1f}')
    print()
    print('Validation Set')
    print('=========================')
    print(f'Number of Tasks: {len(val_data)}')
    if len(val_data) > 0:
        print(f'Mean # of compounds per task: {np.mean([len(i) for i in val_data]):.3f}')
        print(f'Mean # of atoms per compound: {np.mean([i.data.num_nodes/len(i) for i in val_data]):.3f}') 
        print(f'Mean # of bonds per compound: {np.mean([i.data.num_edges/len(i) for i in val_data]):.3f}')
        print(f'Number of atom features: {val_data[0].num_features:.1f}')
        print(f'Number of edge features: {val_data[0].num_edge_features:.1f}')
    print()
    print('Test Set')
    print('=========================')
    print(f'Number of Tasks: {len(test_data)}')
    print(f'Mean # of compounds per task: {np.mean([len(i) for i in test_data]):.3f}') 
    print(f'Mean # of atoms per compound: {np.mean([i.data.num_nodes/len(i) for i in test_data]):.3f}')
    print(f'Mean # of bonds per compound: {np.mean([i.data.num_edges/len(i) for i in test_data]):.3f}')
    print(f'Number of atom features: {test_data[0].num_features:.1f}')
    print(f'Number of edge features: {test_data[0].num_edge_features:.1f}')
    print()

    # initialize embedding function
    model = ProtoNet(train_data[0].num_features, config.out_channels).to(device)
    
    # initialize optimizer
    optim = torch.optim.Adam(params=model.parameters(),
                                lr=config.learning_rate)
    
    # initialize learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                               gamma=config.lr_scheduler_gamma,
                                               step_size=config.lr_scheduler_step)

    # call episodic training loop
    print('Starting training loop...\n')
    res = train(config=config,
                    train=train_data,
                    val=val_data,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler)
    print('\nTraining Complete!\n')

    # save output of training loop
    last_state, best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res

    # test last model
    print('Testing with last model...\n')
    test(config=config,
            test=test_data,
            model=model)

    # load and test best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print('\nTesting with best model...\n')
        test(config=config,
                test=test_data,
                model=model)

    # TODO: write and call visualization functions
    # t-sne!
    
if __name__ == '__main__':
    main()
