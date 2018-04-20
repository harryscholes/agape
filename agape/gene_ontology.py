"""Implements the `GO` class for handling S. pombe gene ontology annotations.
"""
import os
from contextlib import contextmanager
import pandas as pd
from Bio.UniProt.GOA import gafiterator, record_has
from goatools.obo_parser import GODag
from goatools.associations import read_gaf

__all__ = ["GO", "prettify"]

data = os.environ["AGAPEDATA"]


@contextmanager
def go_annotations(file):
    """Handles GO annotation file io.
    """
    with open(file) as f:
        iterator = gafiterator(f)
        yield iterator


class GO(object):
    def __init__(self, *args):
        """Handles S. pombe gene ontology annotations.

        # Arguments
            args: str, sets of GO evidence codes to be included
        """
        self.filters = {
            "taxon": {"Taxon_ID": {"taxon:4896", "taxon:284812"}},
            "protein": {"DB_Object_Type": {"protein"}},
            "experimental": {"Evidence": {"EXP", "IDA", "IPI", "IMP", "IGI",
                                          "IEP", "HDA", "HMP"}},
            "computational": {"Evidence": {"ISS", "ISO", "ISA", "ISM", "IGC",
                                           "IBA", "IBD", "IKR", "IRD", "RCA"}},
            "curated": {"Evidence": {"IC", "TAS"}},
            "automatic": {"Evidence": {"IEA"}},
            "bad": {"Evidence": {"NAS", "ND"}},
        }

        for k, v in self.filters.items():
            setattr(self, k, v)

        self.filter_set = [self.filters[arg] for arg in args] if len(args) > 0 else False

        go_dag = GODag(os.path.join(data, "go.obo"))
        self.go_dag = go_dag
        self.go_id_2_ontology = {k: go_dag[k].namespace for k in go_dag}

    def __iter__(self):
        """Iterate over annotations.
        """
        with go_annotations(os.path.join(data, "gene_association.pombase")) as f:
            for rec in f:
                if rec["GO_ID"] in self.go_dag:  # remove obsolete terms
                    if record_has(rec, self.protein):
                        if self.filter_set:
                            if any([record_has(rec, filt) for filt in self.filter_set]):
                                yield rec
                        else:
                            yield rec

    def get_associations(self, ontology=None):
        """Get associations of gene IDs to GO terms.

        # Arguments
            ontology: str (optional), one of {"biological_process",
                "molecular_function", "cellular_component"}

        # Returns
            dict: maps gene IDs to the GO terms it is annotated them
        """
        associations = read_gaf(os.path.join(data, "gene_association.pombase"))
        all_genes = set(associations.keys())
        wanted_genes = set(rec["DB_Object_ID"] for rec in self)
        unwanted_genes = all_genes - wanted_genes

        associations = remove_unwanted_genes(unwanted_genes, associations)

        if ontology:
            for gene, go_terms in associations.items():
                for go_id in go_terms.copy():
                    if go_id in self.go_dag:
                        if self.go_id_2_ontology[go_id] != ontology:
                            go_terms.remove(go_id)

        self.associations = associations
        return associations

    def get_associations_per_ontology(self):
        """Get the number of annotations per gene in each ontology.

        # Returns
            DataFrame: annotation counts per gene in each ontology.
        """
        go_id_2_ontology_count = {}

        associations = self.get_associations()

        for gene, annotations in associations.items():
            go_id_2_ontology_count[gene] = {}
            d = go_id_2_ontology_count[gene]

            d["biological_process"] = 0
            d["molecular_function"] = 0
            d["cellular_component"] = 0

            for go_id in annotations:
                try:
                    d[self.go_dag[go_id].namespace] += 1
                except KeyError:
                    pass

        df = pd.DataFrame(go_id_2_ontology_count).T
        return df


def remove_unwanted_genes(unwanted_genes, associations):
    """Remove unwanted genes from the set of S. pombe annotations.

    # Arguments
        unwanted_genes: list, list of unwanted genes
        associations: defaultdict, associations read by `read_gaf`
    """
    for gene in unwanted_genes:
        del associations[gene]
    return associations


def prettify(ontology):
    """Pretty print an ontology name.

    # Arguments
        ontology: str, one of {"biological_process", "molecular_function",
            "cellular_component"}

    >>> prettify("biological_process")
    'Biological process'
    """
    ontologies = ("biological_process", "molecular_function", "cellular_component")
    assert ontology in ontologies, f"Must be one of {ontologies}"
    return ontology.capitalize().replace("_", " ")
