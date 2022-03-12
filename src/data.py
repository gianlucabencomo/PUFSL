import os
import re

import torch
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_gz)

from featurize import *


class MoleculeNet(InMemoryDataset):
    """
    Description
    -----------
    Data handler similar to that from torch_geometric. Much of the code
    was left the same but the graph encoding was completely gutted and 
    swapped with the functions from featurize.py.
    """    

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: [display_name, csv_name, smiles_idx, y_idx]
    names = {
        'pcba':     ['PCBA', 'pcba.csv.gz', -1,
                        slice(0, 128)],
        'muv':      ['MUV', 'muv.csv.gz', -1,
                        slice(0, 17)],
        'tox21':    ['Tox21', 'tox21.csv.gz', -1,
                        slice(0, 12)],
        'toxcast':  ['ToxCast', 'toxcast_data.csv.gz', 0, 
                        slice(1, 618)],
            }
    
    def __init__(self, root, name, transform=None, pre_transform=None,
                    pre_filter=None):
        self.name = name.lower()
        assert self.name in self.names.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in dataset:
            # clean line
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')
            
            # load smiles string
            smiles = line[self.names[self.name][2]]

            # convert to Chem molecule and check existence
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # process y
            y = line[self.names[self.name][3]]
            y = y if isinstance(y, list) else [y]
            y = [float(i) if len(i) > 0 else float('NaN') for i in y]
            y = torch.tensor(y, dtype=torch.float).view(1, -1)

            # process nodes (x)
            x = []
            for atom in mol.GetAtoms():
                features = get_atom_features(atom)
                x.append(features)
            
            L = len(x[0])
            x =  torch.tensor(x, dtype=torch.long).view(-1, L)
           
            # process edges
            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                e = get_bond_features(bond)

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]
                
            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            
            L = len(edge_attrs[0])
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, L)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])            


#-------------------------------------------------------------------------------
# Code below is no longer being used, but is being included in this file for the
# time being. Will be deleted once all of the bells and whistles of the lastest
# data handler are implemented.
#-------------------------------------------------------------------------------

import deepchem as dc
from deepchem.molnet.load_function.molnet_loader import TransformerGenerator
from typing import Union, List
import time

def load_data(  dataset: str = 'tox21',
                featurizer: Union[dc.feat.Featurizer, str] = 'ECFP',
                splitter: Union[dc.splits.Splitter, str, None] = None,
                transformers: List[Union[TransformerGenerator, str]] = ['balancing']):
    """
    Description
    ___________
    Small wrapper function to handle the data fetching, cleaning, featurizing, and splitting
    native to the deepchem library.

    Parameters
    __________
    dataset: str
        the dataset to load (tox21, pcba, toxcast, or muv)
    featurizer: Featurizer or str
        the featurizer to use for processing the data.
    splitter: Splitter or str
        the splitter to use for splitting the data into training, validation, and
        test sets.  Alternatively you can pass one of the names from
        dc.molnet.splitters as a shortcut.  If this is None, all the data
        will be included in a single dataset.   
    transformers: list of TransformerGenerators or strings
        the transformations to apply to the data. 

    Returns
    _______
    transformers: list of TransformerGenerators
        list of transformations applied to data
    datasets: train, valid, test or datasets
        DiskDataset consisting of: 
            features (.X) (1D array), 
            labels (.y) (2D array) (labels, task)
            weights (.w) (2D array) (weights, task)
            task_names (.task) (list)                     
    """
    assert (dataset in ['tox21','pcba','toxcast','muv']), 'Specified dataset to load is not recognized.'

    save_dir = './data'
    data_dir = './data'

    if dataset == 'tox21':
        tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=featurizer, splitter=splitter,
                                                            transformers=transformers, data_dir=data_dir,
                                                            save_dir=save_dir)
    elif dataset == 'pcba':
        tasks, datasets, transformers = dc.molnet.load_pcba(featurizer=featurizer, splitter=splitter,
                                                            transformers=transformers, data_dir=data_dir,
                                                            save_dir=save_dir)
    elif dataset == 'toxcast':
        tasks, datasets, transformers = dc.molnet.load_toxcast(featurizer=featurizer, splitter=splitter,
                                                            transformers=transformers, data_dir=data_dir,
                                                            save_dir=save_dir)
    else:
        tasks, datasets, transformers = dc.molnet.load_muv(featurizer=featurizer, splitter=splitter, 
                                                            transformers=transformers, data_dir=data_dir,
                                                            save_dir=save_dir)

    if splitter != None:
        train, valid, test = datasets
        return transformers, train, valid, test 
    else:
        return transformers, datasets  

if __name__ == "__main__":
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True) 
    splitter = dc.splits.RandomSplitter()
    
    # testing
    START = time.time()
    print('Starting to load tox21...')
    transformers, train, valid, test = load_data('tox21', featurizer, splitter) 
    print(f'Loading complete. Load time: {(time.time() - START):.3f} s')

    START = time.time()
    print('Starting to load pcba...')
    transformers, train, valid, test = load_data('pcba', featurizer, splitter) 
    print(f'Loading complete. Load time: {(time.time() - START):.3f} s')

    START = time.time()
    print('Starting to load toxcast...')
    transformers, train, valid, test = load_data('toxcast', featurizer, splitter) 
    print(f'Loading complete. Load time: {(time.time() - START):.3f} s')

    START = time.time()
    print('Starting to load muv...')
    transformers, train, valid, test = load_data('muv', featurizer, splitter) 
    print(f'Loading complete. Load time: {(time.time() - START):.3f} s')
