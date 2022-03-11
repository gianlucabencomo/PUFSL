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
