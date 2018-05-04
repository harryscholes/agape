"""Classes and functions to load data.
"""
import os
import pandas as pd

__all__ = ["Genes", "Biogrid", "STRING"]

data = os.environ["AGAPEDATA"]


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


class Biogrid:
    """Load S. pombe BioGRID database.
    """
    def __init__(self):
        f = "BIOGRID-ORGANISM-Schizosaccharomyces_pombe_972h-3.4.158.tab2.txt"
        df = pd.read_csv(os.path.join(data, f), sep="\t")
        df = df[(df["Organism Interactor A"] == 284812) &
                (df["Organism Interactor B"] == 284812)]
        self.df = df

    def __call__(self, interaction_type=None, graph=False):
        """Call the class instance to filter the loaded interactions.

        # Arguments
            interaction_type: str, {physical, genetic}
            graph: bool, if True return edge list

        # Returns
            DataFrame: BioGRID database

        # Raises
            KeyError: if `interaction_type` is not {physical, genetic}
        """
        df = self.df
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


class STRING:
    """Load S. pombe STRING database.
    """
    def __init__(self):
        f = "4896.protein.links.detailed.v10.5.txt"
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
