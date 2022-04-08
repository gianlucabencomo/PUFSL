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
    
    return parser
