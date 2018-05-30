'''
Preprocess STRING database for use with deepNF.

This script reads the STRING database in $CEREVISIAEDATA and exports six edge
lists of the network's subchannels (excluding text mining) in `.txt` format
for use by `preprocessing.py`.

Usage:
    python STRING.py --output-path $CEREVISIAEDATA/deepNF
'''
import os
import numpy as np
import pandas as pd
from agape.utils import directory_exists, stdout
import argparse


##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-path', default='$CEREVISIAEDATA/deepNF',
                    type=str)
args = parser.parse_args()


######
# io #
######

data = os.environ["CEREVISIAEDATA"]

if directory_exists(args.output_path):
    output_path = os.path.expandvars(args.output_path)


########
# defs #
########

class STRING:
    """Load S. cerevisiae STRING database.
    """
    def __init__(self):
        # f = "4932.protein.links.detailed.v10.5.txt"
        f = "4932.protein.links.detailed.v9.1.txt"
        self.df = pd.read_csv(os.path.join(data, f), sep=" ")
        self.interaction_types = (
            'neighborhood', 'fusion', 'cooccurence', 'coexpression',
            'experimental', 'database')

    def get(self, interaction_type=None):
        """Call the class instance to filter the loaded interactions.

        # Arguments
            interaction_type: str, {neighborhood, fusion, cooccurence,
                coexpression, experimental, database}

        # Returns
            DataFrame: STRING database

        # Raises
            KeyError: if `interaction_type` is not in {neighborhood, fusion,
                cooccurence, coexpression, experimental, database}
        """
        if all((interaction_type is not None,
                interaction_type not in self.interaction_types)):

            raise KeyError(
                f"`interaction_type` must be one of: {self.interaction_types}")

        return self.df[["protein1", "protein2", interaction_type]]

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
                                   f"cerevisiae_string_{col}_adjacency.txt"),
                      sep="\t", index=False, header=False)

        pd.Series(self.d).to_csv(os.path.join(output_path,
                                              'cerevisiae_net_genes.csv'))


def main():
    print(__doc__)
    stdout("Command line arguments", args)
    s = STRING()
    s.convert_ids_to_numbers()
    stdout('Writing networks to', output_path)
    s.write()


if __name__ == "__main__":
    main()
