'''
Make ageing gene targets data set.

Usage:
    python ageing_genes.py
'''
import os
import pandas as pd
from agape.utils import directory_exists
import argparse
import numpy as np
from scipy import sparse, io as sio


##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-path',
                    default='$AGAPEDATA/ageing_genes', type=str)
args = parser.parse_args()


output_path = directory_exists(args.output_path)


def sparse_colvector(filename):
    string_indices = pd.read_csv(
        os.path.join(os.path.expandvars('$AGAPEDATA'),
                     'deepNF', 'networks', 'yeast_net_genes.csv'), header=None)

    df = pd.read_csv(
        os.path.join(os.path.expandvars('$AGAPEDATA'),
                     'ageing_genes', filename))

    df['Systematic ID'] = df['Systematic ID'].apply(lambda x: f'4896.{x}.1')

    string_proteins = string_indices[0].values
    df = df[df['Systematic ID'].isin(string_proteins)]

    mapping = dict(zip(string_indices[0], string_indices[1]))
    df['Systematic ID'] = df['Systematic ID'].apply(lambda x: mapping[x])

    i = list(df['Systematic ID'])
    j = [0] * len(i)
    A = sparse.coo_matrix((np.tile(1, len(i)), (i, j)),
                          shape=(len(string_proteins), 1))
    return A


FYPO_IDs = ['FYPO_0001309.tsv', 'FYPO_0004344.tsv']

d = {k[:-4]: sparse_colvector(k) for k in FYPO_IDs}

sio.savemat(os.path.join(output_path, "ageing_genes.mat"), d)
