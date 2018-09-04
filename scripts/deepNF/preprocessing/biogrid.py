'''
Preprocess BioGRID genetic interactions for use with deepNF.

Usage:
    python biogrid.py
'''
import os
import pandas as pd
from agape.load import Biogrid
from agape.utils import directory_exists
import argparse


##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-path',
                    default='$AGAPEDATA/deepNF/networks', type=str)
args = parser.parse_args()


output_path = directory_exists(args.output_path)

string_indices = pd.read_csv(
    os.path.join(os.path.expandvars('$AGAPEDATA'),
                 'deepNF', 'networks', 'yeast_net_genes.csv'), header=None)

df = Biogrid()('genetic', graph=True)

for c in ('source', 'target'):
    df[c] = df[c].apply(lambda x: f'4896.{x}.1')

string_proteins = string_indices[0].values
df = df[(df.source.isin(string_proteins)) & (df.target.isin(string_proteins))]

mapping = dict(zip(string_indices[0], string_indices[1]))

for c in ('source', 'target'):
    df[c] = df[c].apply(lambda x: mapping[x])

df = df.sort_values(by=['source', 'target'])

df['weight'] = 1.

df.to_csv(os.path.join(output_path,
                       f"yeast_biogrid_genetic_adjacency.txt"),
          sep="\t", index=False, header=False)
