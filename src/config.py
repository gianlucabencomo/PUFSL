import os
import argparse
import numpy as np

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, 
                        choices=['tox21', 'pcba', 'toxcast_data', 'muv', 'sider'],
                        required=True,
                        help='Dataset to train/test on.')
    parser.add_argument('-s', '--seed', type=int, 
                        default=np.random.randint(0, 2**32 - 1),
                        help='Specify a manual seed. Otherwise random.')    
    parser.add_argument('-dp', '--data_path', type=str,
                        default=os.path.join(os.getcwd(), 'data'), 
                        help='Data path to save to.') 
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=0.001,
                        help='Learning rate, default=0.001.')    
    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=50',
                        default=50)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)
    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='Root where to store models, losses and accuracies',
                        default='output')    
    parser.add_argument('-ep', '--epochs',
                        type=int,
                        help='Number of epochs to train for, default=100.',
                        default=100) 
    parser.add_argument('-it', '--iterations',
                        type=int,
                        help='Number of iterations for each training episode/epoch.',
                        default=1000)
    parser.add_argument('-trsp', '--tr_split',
                        type=float,
                        help='Percentage of tasks to use for training set, default=0.8.',
                        default=0.8)
    parser.add_argument('-vsp', '--v_split',
                        type=float,
                        help='Percentage of tasks to use for validation set, default=0.0',
                        default=0.0)
    parser.add_argument('-tesp', '--te_split',
                        type=float,
                        help='Percentage of tasks to use for test set, default=0.2',
                        default=0.2)
    parser.add_argument('-tr', '--transform',
                        help='torch_geometric.transform object that can modify and customize data object',
                        default=None)
    parser.add_argument('-ptr', '--pre_transform',
                        help='Pre-transformation. (torch_geometric.transform object)',
                        default=None)
    parser.add_argument('-pf', '--pre_filter',
                        help='Pre-filter to apply to data.',
                        default=None)
    parser.add_argument('-oc', '--out_channels',
                        type=int,
                        help='Number of channels in output of GNN.',
                        default='65')
    parser.add_argument('-ps', '--pos_support',
                        type=int,
                        help='Number of positive support examples',
                        default=10) 
    parser.add_argument('-pq', '--pos_query',
                        type=int, 
                        help='Number of positive query examples',
                        default=10) 
    parser.add_argument('-ns', '--neg_support', 
                        type=int,
                        help='Number of negative support examples',
                        default=10) 
    parser.add_argument('-nq', '--neg_query', 
                        type=int,
                        help='Number of negative query examples',
                        default=10) 
    parser.add_argument('-tpq', '--test_pos_query',
                        type=int,
                        help='Number of positive query examples to use during testing.',
                        default=200)
    parser.add_argument('-tnq', '--test_neg_query',
                        type=int,
                        help='Number of negative query examples to use during testing.',
                        default=200)

    return parser
