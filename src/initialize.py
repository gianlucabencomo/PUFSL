import numpy as np
import torch
from data import *

lengths =   {
            'tox21': 12,
            'pcba': 128,
            'toxcast_data': 618,
            'muv': 17,
            'sider': 28
            }

def init_seed(seed):
    '''
    Initialize seed.
    ''' 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_data(config):
    '''
    Based on configs, returns train, val, and test datasets.
    First n tasks are train, last m - n tasks are test.
    '''    
    train, val, test = [], [], []
    for i in range(0, lengths[config.dataset]):
        if i < int(lengths[config.dataset] * config.tr_split):
            train.append(MoleculeNet(root=config.data_path, name=config.dataset,
                                    task=i, 
                                    transform=config.transform, 
                                    pre_transform=config.pre_transform, 
                                    pre_filter=config.pre_filter))
        elif i < int(lengths[config.dataset] * (config.tr_split + config.v_split)):        
            val.append(MoleculeNet(root=config.data_path, name=config.dataset,
                                    task=i,
                                    transform=config.transform, 
                                    pre_transform=config.pre_transform, 
                                    pre_filter=config.pre_filter))
        else:
            test.append(MoleculeNet(root=config.data_path, name=config.dataset,
                                    task=i,
                                    transform=config.transform, 
                                    pre_transform=config.pre_transform, 
                                    pre_filter=config.pre_filter))
                    
    return train, val, test
