import os
import numpy as np
import pandas as pd
import json
from Bio.PDB import *

parse = PDB.PDBParser()

def get_data(data_path):
    X = []
    for file in os.listdir(data_path):
        if file.split('.')[-1] == 'pdb':
            structure = parse.get_structure(id=file.split('.')[0], file=file)
            dict_ = {
                    'coords':{'CA':[], 'O':[], 'N':[], 'C':[]}
                    }
            for res in structure.get_residues():
                if res['CA']!=None and res['O']!=None and res['N']!=None and res['C']!=None:
                    dict_['coords']['CA'].append(list(res['CA'].get_vector()))
                    dict_['coords']['O'].append(list(res['O'].get_vector()))
                    dict_['coords']['N'].append(list(res['N'].get_vector()))
                    dict_['coords']['C'].append(list(res['C'].get_vector()))
            X.append(dict_)
    return X

def save_data(X, save_path)
    with open(save_path, "w") as data:
        json.dump(X, data)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Dataset')

    parser.add_argument('--data_path', default="", type=str)
    parser.add_argument('--save_path', default="data.json", type=str)

    args = parser.parse_args("")
    
    X = get_data(args.data_path)
    save_data(X, args.save_path)
