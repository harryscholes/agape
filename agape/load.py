"""Classes and functions to load data.
"""
import os
import pandas as pd

__all__ = ["Genes", "biogrid"]

data = os.environ["POMBAGEDATA"]


class Genes(object):
    """Load S. pombe gene IDs.
    """
    def __init__(self):
        df = pd.read_csv(os.path.join(data,
                                      "schizosaccharomyces_pombe.genome.gff3"),
                         skiprows=6,
                         header=None,
                         sep="\t")
        self.df = df.copy()
        df = df[df[2] == "gene"]
        _, df[8] = df[8].str.split("ID=gene:").str
        df[8], _ = df[8].str.split(";Name=").str
        self.genes = list(df[8].unique())

    def viability(self, phenotype=None):
        """Get genes annotated with a viability phenotype.

        # Arguments
            phenotype: str (optional), {viable, inviable, condition-dependent}
        """
        if not hasattr(self, "viability_df"):
            df = pd.read_csv(os.path.join(data, "FYPOviability.tsv"), sep="\t",
                             header=None, names=["GeneID", "Phenotype"])

            self.viability_df = df

        if phenotype is None:
            return self.viability_df

        phenotypes = self.viability_df.Phenotype.unique()

        if phenotype not in phenotypes:
            raise KeyError(f"`phenotype` must be one of:", phenotypes)

        df = self.viability_df
        df = df[df.Phenotype == phenotype]
        return list(df.GeneID)


def biogrid(interaction_type=None, graph=False):
    """Load S. pombe BioGRID database.

    # Arguments
        interaction_type: str, {physical, genetic}
        graph: bool, if True return edge list

    # Returns
        DataFrame: BioGRID database

    # Raises
        KeyError: if `interaction_type` is not {physical, genetic}
    """
    f = "BIOGRID-ORGANISM-Schizosaccharomyces_pombe_972h-3.4.158.tab2.txt"
    df = pd.read_csv(os.path.join(data, f), sep="\t")

    df = df[(df["Organism Interactor A"] == 284812) &
            (df["Organism Interactor B"] == 284812)]

    if interaction_type is not None:

        interaction_types = df["Experimental System Type"].unique()

        if interaction_type not in interaction_types:
            raise KeyError(("`interaction_type` must be one of:",
                            f"{', '.join(interaction_types)}"))

        df = df[df["Experimental System Type"] == interaction_type]

    if graph:
        df = df[['Systematic Name Interactor A',
                 'Systematic Name Interactor B']]
        df.columns = ["source", "target"]

    return df
