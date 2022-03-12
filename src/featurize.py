import numpy as np

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import torch

# the following functions were inspired by a blog post made by Markus Dablander of the 
# Oxford Protein Informatics Group

def one_hot_encoding(x: str, permitted: list):
    
    # maps unrecognized values to the last element, which contains 'unknown'
    if x not in permitted:
        x = permitted[-1]
    
    # encodes list where index = 1 if element occurs at that index
    encoding = list(map(lambda s: int(x == s), permitted))
    
    return encoding


def get_atom_features(  atom, 
                        use_chirality: bool = True, 
                        hydrogens_implicit: bool = True):
    
    # list of permitted atoms
    permitted_atoms = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al',
                       'I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 
                       'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']

    # add hydrogens if hydrogens_implicit == False
    if not hydrogens_implicit:
        permitted_atoms.insert(0, 'H')

    # compute the one hot encoding for the atom's element
    a_type = one_hot_encoding(str(atom.GetSymbol()), permitted_atoms)

    # compute the number of heavy neighbors
    a_heavy = one_hot_encoding(int(atom.GetDegree()), [range(0, 5), "5+"])

    # compute the atom's formal charge
    a_charge = one_hot_encoding(int(atom.GetFormalCharge()), [range(-5, 7), "Extreme"])

    # specify the atom's hybridisation
    a_hybridisation = one_hot_encoding(str(atom.GetHybridization()), 
                                        ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

    # bool: is the atom in a ring?
    a_ring = [int(atom.IsInRing())]

    # bool: is the atom aromatic?
    a_aromatic = [int(atom.GetIsAromatic())]

    # get the atomic mass and normalize
    a_mass = [float((atom.GetMass() - 10.812)/116.092)]

    # compute the van der waals radius
    a_vdw_radius = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]

    # compute the covalent radius
    a_cov_radius = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

    # create a feature vector by concat. everything above
    features = a_type + a_heavy + a_charge + a_hybridisation + a_ring + a_aromatic + a_mass + a_vdw_radius + a_cov_radius

    # chirality features if true
    if use_chirality == True:
        a_chirality = one_hot_encoding(str(atom.GetChiralTag()), 
                                        ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", 
                                         "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        features += a_chirality
    
    # add hydrogen info if true
    if hydrogens_implicit == True:
        a_hydrogens = one_hot_encoding(int(atom.GetTotalNumHs()), [range(0,9), "9+"])
        features += a_hydrogens
    
    return features


def get_bond_features(  bond, 
                        use_stereochemistry: bool = True):
      
    # permitted list of bonds
    permitted_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                       Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    
    # compute the one hot encoding for the type of bond
    b_type = one_hot_encoding(bond.GetBondType(), permitted_bonds)
    
    # is the bond conjugated?
    b_conj = [int(bond.GetIsConjugated())]
    
    # is the bond in a ring?
    b_ring = [int(bond.IsInRing())]

    # create a feature vector
    features = b_type + b_conj + b_ring

    # add stereochemistry if true
    if use_stereochemistry == True:
        b_stereo = one_hot_encoding(str(bond.GetStereo()), ['STEREONONE',
                                                            'STEREOZ',
                                                            'STEREOE',
                                                            'STEREOCIS',
                                                            'STEREOTRANS',
                                                            'STEREOANY',
                                                            ])
        features += b_stereo

    return features



