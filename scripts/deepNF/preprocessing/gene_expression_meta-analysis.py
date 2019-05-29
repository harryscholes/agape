'''
Preprocess BioGRID genetic interactions for use with deepNF.

Usage:
    python biogrid.py
'''
import os
import pandas as pd
from agape.utils import directory_exists
import argparse
from sklearn.preprocessing import minmax_scale


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

df = pd.read_csv(os.path.join(os.path.expandvars('$AGAPEDATA'),
                              'pombeallpairs..genexp.txt'), sep='\t')
df = df.dropna()

cols = ['Gene1', 'Gene2']

for c in cols:
    df[c] = df[c].apply(lambda x: f'4896.{x}.1')

string_proteins = string_indices[0].values
df = df[(df.Gene1.isin(string_proteins)) & (df.Gene2.isin(string_proteins))]

mapping = dict(zip(string_indices[0], string_indices[1]))

for c in cols:
    df[c] = df[c].apply(lambda x: mapping[x])

df = df.sort_values(by=cols)

df.Expression_correlation = minmax_scale(df.Expression_correlation)

df.to_csv(os.path.join(output_path,
                       f"yeast_z_gene_expression_meta-analysis_adjacency.txt"),
          sep="\t", index=False, header=False)
