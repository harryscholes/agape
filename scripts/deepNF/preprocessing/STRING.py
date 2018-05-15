'''
Preprocess STRING database for use with deepNF.

This script reads the STRING database in $AGAPEDATA and exports six edge
lists of the network's subchannels (excluding text mining) in `.txt` format
for use by `preprocessing.py`.

Usage:
    python STRING.py --output-path $AGAPEDATA/deepNF
'''
import os
import numpy as np
import pandas as pd
from agape.load import STRING
from agape.utils import directory_exists, stdout
import argparse


##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-path', default='$AGAPEDATA/deepNF',
                    type=str)
args = parser.parse_args()


######
# io #
######

if directory_exists(args.output_path):
    output_path = args.output_path


########
# defs #
########

class STRING_deepNF(STRING):
    def __init__(self):
        super().__init__()

    def convert_ids_to_numbers(self):
        nodes = np.unique(self.df[["protein1", "protein2"]].values.flatten())
        nodes.sort()
        # Dict mapping gene IDs to ints
        self.d = {name: number for name, number
                  in zip(nodes, range(len(nodes)))}
        # Relabel nodes
        for col in ("protein1", "protein2"):
            self.df[col] = [self.d[i] for i in self.df[col]]

    def write(self):
        for col in self.interaction_types:
            print(f"    {col}")
            df = self.get(col)
            df = df[df[col] != 0]
            df.loc[:, col] /= 1000
            df.to_csv(os.path.join(output_path,
                                   f"yeast_string_{col}_adjacency.txt"),
                      sep="\t", index=False, header=False)

        pd.Series(self.d).to_csv(os.path.join(output_path,
                                              'yeast_net_genes.csv'))


def main():
    print(__doc__)
    stdout("Command line arguments", args)
    s = STRING_deepNF()
    s.convert_ids_to_numbers()
    stdout('Writing networks to', output_path)
    s.write()


if __name__ == "__main__":
    main()
