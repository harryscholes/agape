"""Implements the `GO` class for handling S. pombe gene ontology annotations.
"""
import os
from contextlib import contextmanager
from Bio.UniProt.GOA import gafiterator, record_has
from goatools.obo_parser import GODag
from goatools.associations import read_gaf
from collections import defaultdict
from .exceptions import GeneOntologyError
import copy
from .base import Base

__all__ = ["GO", "prettify"]

data = os.environ["AGAPEDATA"]


@contextmanager
def go_annotations(filepath: str):
    """Handles GO annotation file io.

    # Arguments
        filepath: str, filepath of GO annotation file

    # Returns
        gafiterator

    # Raises
        FileNotFoundError
    """
    try:
        with open(filepath) as f:
            iterator = gafiterator(f)
            yield iterator
    except FileNotFoundError as err:
        raise


class GO(Base):
    """Handles S. pombe gene ontology annotations.

    # Arguments
        args: str, sets of GO evidence codes to be included
    """
    def __init__(self, *allowed_evidence_codes,
                 go_dag_path='$AGAPEDATA/go.obo',
                 associations_path='$AGAPEDATA/gene_association.pombase'):
        super().__init__()

        self.set_evidence_codes()
        self.set_allowed_evidence_codes(allowed_evidence_codes)
        self.go_dag_path = os.path.expandvars(go_dag_path)
        self.associations_path = os.path.expandvars(associations_path)

    def set_evidence_codes(self):
        self.evidence_codes = {
            "taxon": {
                "Taxon_ID": {
                    "taxon:4896", "taxon:284812"}},
            "protein": {
                "DB_Object_Type": {
                    "protein"}},
            "experimental": {
                "Evidence": {
                    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HDA", "HMP"}},
            "computational": {
                "Evidence": {
                    "ISS", "ISO", "ISA", "ISM", "IGC", "IBA", "IBD", "IKR",
                    "IRD", "RCA"}},
            "curated": {
                "Evidence": {
                    "IC", "TAS"}},
            "automatic": {
                "Evidence": {
                    "IEA"}},
            "bad": {
                "Evidence": {
                    "NAS", "ND"}}}

        for k, v in self.evidence_codes.items():
            setattr(self, k, v)

    def set_allowed_evidence_codes(self, allowed_evidence_codes):
        """Set which evidence code sets to include when iterating over `GO`.

        # Arguments
            allowed_evidence_codes: list, keys of `self.evidence_codes`
        """
        try:
            self.allowed_evidence_codes = [self.evidence_codes[i] for i in
                                           allowed_evidence_codes] \
                if len(allowed_evidence_codes) > 0 else None
        except KeyError as err:
            raise GeneOntologyError(
                f"Not a valid evidence code set: {err.args[0]}")

    def load_go_dag(self):
        """Load GO DAG.

        # Arguments
            filepath: str (optional), path to go.obo
        """
        filepath = self.go_dag_path
        if not os.path.exists(filepath):
            raise GeneOntologyError(f"{os.path.basename(filepath)} does not exist at {os.path.dirname(filepath)}")

        go_dag = GODag(filepath)
        self.go_dag = go_dag

    def set(self, name, value):
        setattr(self, name, value)

    def get(self, name):
        return getattr(self, name)

    def __iter__(self):
        """Iterate over annotations.
        """
        filepath = self.associations_path
        if not os.path.exists(filepath):
            raise GeneOntologyError(f"{os.path.basename(filepath)} does not exist at {os.path.dirname(filepath)}")

        if not hasattr(self, "go_dag"):
            self.load_go_dag()

        with go_annotations(filepath) as f:
            for rec in f:
                # Remove obsolete terms
                if rec["GO_ID"] in self.go_dag:
                    # Only iterate over proteins
                    if record_has(rec, self.protein):
                        # Only iterate over allowed evidence codes
                        if self.allowed_evidence_codes is not None:
                            # Yield annotation if it has an allowed evidence code
                            if any([record_has(rec, ec) for ec in self.allowed_evidence_codes]):
                                yield rec
                        # Otherwise iterate over all evidence codes
                        else:
                            yield rec

    def get_associations(self, ontology=None):
        """Get associations of gene IDs to GO terms.

        Ontologies: P = biological process, F = molecular function,
            C = cellular component

        # Arguments
            ontology: str (optional), one of {"P", "F", "C"}

        # Returns
            dict: maps gene IDs to the GO terms it is annotated them

        # Raises
            GeneOntologyError: if `ontology` is not valid
        """
        if ontology is not None and ontology not in ("P", "F", "C"):
            raise GeneOntologyError(f"Not a valid ontology: {ontology}")

        # Load a defaultdict mapping gene_ids to the GO terms annotated to them
        if not hasattr(self, "all_associations"):
            self.all_associations = read_gaf(self.associations_path)

        all_associations = copy.deepcopy(self.all_associations)

        # Remove genes that do not have any annotations with an accepted
        # evidence code
        wanted_genes = set(rec["DB_Object_ID"] for rec in self)
        associations = self.remove_unwanted_genes(wanted_genes, all_associations)
        # Only consider GO terms from a particular ontology
        if ontology is not None:
            # term2ontology_dict = self.term2ontology()
            d = self.ontology2term()
            accepted_terms = d[ontology]
            # Iterate over dictionary of associations
            for gene, go_terms in associations.items():
                for go_id in go_terms.copy():
                    # Remove obsolete terms
                    if go_id in self.go_dag:
                        # Remove GO terms from other ontologies
                        if go_id not in accepted_terms:
                            go_terms.remove(go_id)

        self.associations = associations
        return associations

    def remove_unwanted_genes(self, wanted_genes, associations):
        """Remove unwanted genes from the set of annotations.

        # Arguments
            wanted_genes: set, genes to keep in `associations`
            associations: defaultdict, associations read by `read_gaf` mapping
                gene_ids to the GO terms annotated to them
        """
        # Get `unwanted_genes` that do not have any annotations with an
        # accepted evidence code
        all_genes = set(associations)
        unwanted_genes = all_genes - set(wanted_genes)

        # Delete the unwanted genes from `associations`
        for gene in unwanted_genes:
            del associations[gene]
        return associations

    def term2ontology(self) -> dict:
        """Maps GO terms to their ontology.
        """
        if not hasattr(self, "term2ontology_d"):
            d = {rec["GO_ID"]: rec["Aspect"] for rec in self}
            self.term2ontology_d = d
        return self.term2ontology_d

    def ontology2term(self) -> defaultdict:
        """Maps ontologies to the GO terms in that ontology.
        """
        if not hasattr(self, "term2term2ontology_d"):
            self.term2ontology()

        d_r = defaultdict(set)
        for go_id, ontology in self.term2ontology_d.items():
            d_r[ontology].add(go_id)
        self.ontology2term_d = d_r
        return d_r


def prettify(ontology: str) -> str:
    """Pretty print an ontology name.

    # Arguments
        ontology: str, one of {"biological_process", "molecular_function",
            "cellular_component"}

    >>> prettify("biological_process")
    'Biological process'
    """
    ontologies = ("biological_process", "molecular_function",
                  "cellular_component")
    if ontology not in ontologies:
        raise GeneOntologyError(f"Must be one of {ontologies}")
    return ontology.capitalize().replace("_", " ")
