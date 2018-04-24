from pytest import raises
from agape.gene_ontology import remove_unwanted_genes, prettify


class TestPrettify(object):
    def test_prettify(self):
        assert prettify("biological_process") == "Biological process"

    def test_raises_AssertionError(self):
        with raises(AssertionError):
            prettify("NOTVALID")


class TestRemoveUnwantedGenes(object):
    def setup_method(self):
        self.d = {"a": 1, "b": 2, "c": 3}

    def test_remove(self):
        d = self.d
        uw = ["a", "b"]
        assert remove_unwanted_genes(uw, d) == {"c": 3}

    def test_raises_KeyError(self):
        d = self.d
        uw = ["NOTAKEY"]
        with raises(KeyError):
            remove_unwanted_genes(uw, d)
