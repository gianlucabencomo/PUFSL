import numpy as np
import torch

def init_seed(seed):
    '''
    Initialize seed.
    ''' 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
