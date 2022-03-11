import time
import click
import torch
import deepchem as dc
import numpy as np
from data import load_data
from deepchem.models import GATModel, GCNModel, AttentiveFPModel


# this script is intended to run all the features of the deepchem library used in this project, 
# without any fancy parlor tricks. 

@click.command()
@click.option('-d', '--dataset',
        type=click.Choice(['tox21', 'pcba', 'toxcast', 'muv']),
        required=True,
        help='Dataset to train model on.')
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
@click.option('-u', '--use_edges', default=True,
        help='Use edges in molecular graph.')
@click.option('-sp', '--save_path', 
        required=True,
        help='Specify the save path to which models/results should be saved.')

def simple( dataset: str,
            model: str,
            epochs: int,
            learning_rate: float,
            weight_decay: float,
            batch_size: int,
            use_edges: bool,
            seed: int,
            save_path: str):
            
    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # featurizer setup
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=use_edges)

    # configure spliter
    splitter = dc.splits.RandomSplitter()

    # load data
    START = time.time()
    transformers, train, valid, test = load_data(dataset, featurizer, splitter)
    print(f'Loading complete. Load time: {(time.time() - START):.3f} s')

    # set metric for evaluating the validation set
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    n_tasks = len(train.tasks)

    if learning_rate != 0.0:
        learning_rate = dc.models.optimizers.ExponentialDecay(learning_rate, 0.9, 1000)
    
    if model == "GAT":
        model = GATModel(mode='classification', n_tasks=n_tasks, batch_size=batch_size, 
                            learning_rate=learning_rate, device=device, model_dir=save_path)
    elif model == "GCN":
        model = GCNModel(mode='classification', n_tasks=n_tasks, batch_size=batch_size, 
                            learning_rate=learning_rate, device=device, model_dir=save_path)   
    else:
        model = AttentiveFPModel(mode='classification', n_tasks=n_tasks, batch_size=batch_size, 
                            learning_rate=learning_rate, device=device, model_dir=save_path)
    
    callback = dc.models.ValidationCallback(valid, 1000, metric)
    
    print('Model loaded.  Beginning to train...')
    START = time.time()
    loss = model.fit(train, nb_epoch=epochs, callbacks=callback)
    print(f'Training complete. Train time: {(time.time() - START):.3f} s')
    
    

if __name__ == "__main__":
    simple()
