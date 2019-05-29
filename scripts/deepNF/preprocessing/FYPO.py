'''
Preprocess FYPO annotations for use with deepNF.

Usage:
    python FYPO.py
'''
import os
import pandas as pd
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

df = pd.read_csv(os.path.join(os.path.expandvars('$AGAPEDATA'),
                              'phenotype_annotations.pombase.phaf'), sep='\t')
df = df[['Gene systematic ID', 'FYPO ID']]

df['Gene systematic ID'] = df['Gene systematic ID'].apply(
    lambda x: f'4896.{x}.1')

string_proteins = string_indices[0].values
df = df[df['Gene systematic ID'].isin(string_proteins)]

gene_mapping = dict(zip(string_indices[0], string_indices[1]))

df['Gene systematic ID'] = df['Gene systematic ID'].apply(
    lambda x: gene_mapping[x])

FYPO_IDs = sorted(df['FYPO ID'].unique())
fypo_mapping = dict(zip(FYPO_IDs, range(len(FYPO_IDs))))

df['FYPO ID'] = df['FYPO ID'].apply(lambda x: fypo_mapping[x])

df = df.drop_duplicates()
df = df.sort_values(by=['Gene systematic ID', 'FYPO ID'])

df['weight'] = 1.

df.to_csv(os.path.join(output_path,
                       f"yeast_z_fypo_adjacency.txt"),
          sep="\t", index=False, header=False)
