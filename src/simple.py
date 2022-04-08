import time
import click
import torch
import numpy as np

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader

from data import *

# this script is intended to run all the features of the deepchem library used in this project, 
# without any fancy parlor tricks. 


@click.command()
@click.option('-d', '--dataset',
        type=click.Choice(['tox21', 'pcba', 'toxcast_dataset', 'muv']),
        required=True,
        help='Dataset to train model on.')
@click.option('-t', '--task', default=0, 
        help='Task to train on.')
@click.option('-m', '--model',
        type=click.Choice(['GAT', 'GCN', 'AFP']),
        required=True,
        help='Model : [GATModel, GCNModel, AttentiveFPModel].')
@click.option('-lr', '--learning_rate', default=1e-3,
        help='Initial learning rate.')
@click.option('-e', '--epochs', default=50, 
        help='Number of training epochs.')
@click.option('-b', '--batch_size', default=16,
        help='Batch size for both training and validation.')
@click.option('-wd', '--weight_decay', default=1e-3,
        help='Weight decay.')
@click.option('-s', '--seed', default=np.random.randint(10000),
        help='Seed for train/test split.')
@click.option('-uc', '--use_chirality', is_flag=True,
        help='Include chirality in the feature vector.')
@click.option('-hi', '--hydrogens_implicit', is_flag=True,
        help='Treat hydrogens implicitly when generating the feature vector (recommended).')
@click.option('-us', '--use_stereochemistry', is_flag=True,
        help='Include stereochemistry info when generating the bond feature vectors.')
@click.option('-sp', '--save_path', 
        required=True,
        help='Specify the save path to which models/results should be saved.')


def simple( dataset: str,
            task: int,
            model: str,
            epochs: int,
            learning_rate: float,
            weight_decay: float,
            batch_size: int,
            use_chirality: bool, # not being used right now
            hydrogens_implicit: bool, # not being used right now
            use_stereochemistry: bool, # not being used right now
            seed: int,
            save_path: str,
            transform = None, # change later ... keeping None for now for simplicity
            pre_transform = None, # change later ... keeping None for now for simplicity
            pre_filter = None): # change later ... keeping None for now for simplicity
    
    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    dataset = MoleculeNet(root=os.path.join(os.getcwd(), 'data'), name=dataset, 
                        transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    print()
    print(f'Dataset: {dataset.name}')
    print('====================')
    print(f'Number of graphs (compounds): {len(dataset)}')
    print(f'Number of features per atom: {dataset.num_features}')
    print(f'Number of features per bond: {dataset.num_edge_features}')
    print(f'Number of tasks: {dataset.num_classes}') 
    print(f'Average number of atoms per molecule: {len(dataset.data.x) / len(dataset):.2f}')
    print(f'Average number of bonds per molecule: {len(dataset.data.edge_attr) / len(dataset):.2f}')
    
    
    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    assert 0 <= task and task < dataset.num_classes, f'Selected task is out of range (0 <= task < {dataset.num_classes})'
    dataset.data.y = dataset.data.y[:,task]
    
    # shuffle dataset
    dataset = dataset.shuffle()  

    # train / val / test split
    factor = int(0.1 * len(dataset))

    train = dataset[:factor*8]
    valid = dataset[factor*8:factor*9]
    test = dataset[factor*9:]
    
    print()
    print(f'Selected Task: {task}')
    print('=============================================================')
    print(f'Number of training graphs: {len(train)}')
    print(f'Number of validation graphs: {len(valid)}')
    print(f'Number of test graphs: {len(test)}')       

    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)    
    
    # TODO: implement training + eval
   
if __name__ == "__main__":
    simple()
