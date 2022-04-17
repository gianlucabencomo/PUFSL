import os
import argparse
import numpy as np

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, 
                        choices=['tox21', 'pcba', 'toxcast_dataset', 'muv'],
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
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)

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
    return parser
