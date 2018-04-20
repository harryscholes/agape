from pytest import raises
from agape.gene_ontology import prettify


class TestPrettify(object):
    def test_prettify(self):
        assert prettify("biological_process") == "Biological process"

    def test_raises_AssertionError(self):
        with raises(AssertionError):
            prettify("NOTVALID")
