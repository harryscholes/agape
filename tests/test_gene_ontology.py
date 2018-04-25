from pytest import raises, mark
from agape.gene_ontology import GO, prettify
from agape.exceptions import GeneOntologyError
import os


class TestGO(object):
    def setup_method(self):
        self.obj = GO()

    def test_evidence_codes(self):
        assert len(self.obj.evidence_codes) == 7

    def test_evidence_codes_attrs(self):
        for ec in self.obj.evidence_codes:
            assert hasattr(self.obj, ec)

    def test_set_allowed_evidence_codes(self):
        a = ["curated", "automatic"]
        expected = [{"Evidence": {"IC", "TAS"}}, {"Evidence": {"IEA"}}]
        self.obj.set_allowed_evidence_codes(a)
        assert self.obj.allowed_evidence_codes == expected

    def test_set_allowed_evidence_codes_raises_GeneOntologyError(self):
        with raises(GeneOntologyError):
            self.obj.set_allowed_evidence_codes(["NOTAKEY"])

    def test_set(self):
        name = "TESTNAME"
        value = "TESTVALUE"
        self.obj.set(name, value)
        assert hasattr(self.obj, name)
        assert getattr(self.obj, name) == value

    def test_get(self):
        name = "TESTNAME"
        value = "TESTVALUE"
        setattr(self.obj, name, value)
        assert hasattr(self.obj, name)
        assert self.obj.get(name) == value

    @mark.io
    def test_load_go_dag(self):
        self.obj.load_go_dag()
        assert hasattr(self.obj, "go_dag")

    def test_load_go_dag_raises_GeneOntologyError(self):
        with raises(GeneOntologyError):
            self.obj.load_go_dag("NOTAPATH")

    def test_iter_raises_GeneOntologyError(self):
        self.obj.set("custom_association_file_path", "NOTAPATH")
        iterator = iter(self.obj)
        with raises(GeneOntologyError):
            next(iterator)

    def test_remove_unwanted_genes(self):
        d = {"a": 1, "b": 2, "c": 3}
        w = set(("c"))
        assert self.obj.remove_unwanted_genes(w, d) == {"c": 3}


class TestPrettify(object):
    def test_prettify(self):
        assert prettify("biological_process") == "Biological process"

    def test_raises_GeneOntologyError(self):
        with raises(GeneOntologyError):
            prettify("NOTVALID")
