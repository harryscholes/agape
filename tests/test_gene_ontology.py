from pytest import raises, fixture
from agape.gene_ontology import GO, prettify, go_annotations
from agape.exceptions import GeneOntologyError
import inspect
import os
from collections import defaultdict
import itertools


class TestGo_Annotations:
    def setup_method(self):
        filepath = os.path.join(os.environ["AGAPEDATA"],
                                "gene_association.pombase")
        self.obj = go_annotations(filepath)

    def test_context_manager(self):
        with self.obj as o:
            assert inspect.isgenerator(o)

    def test_context_manager_raises_FileNotFoundError(self):
        with raises(FileNotFoundError):
            with go_annotations("NOTAFILE") as o:
                pass


@fixture(scope="module")
def GoObj():
    """Instantiate `GO` object with `go_dag`.
    """
    go = GO()
    go.load_go_dag()
    return go


class TestGO:
    def setup_method(self):
        self.obj = GO()
        self.example_evidence_codes = ["curated", "automatic"]

    def test_evidence_codes(self, GoObj):
        assert len(GoObj.evidence_codes) == 7

    def test_evidence_codes_attrs(self, GoObj):
        for ec in GoObj.evidence_codes:
            assert hasattr(GoObj, ec)

    def test_set_allowed_evidence_codes(self, GoObj):
        try:
            expected = [{"Evidence": {"IC", "TAS"}}, {"Evidence": {"IEA"}}]
            GoObj.set_allowed_evidence_codes(self.example_evidence_codes)
            assert GoObj.allowed_evidence_codes == expected
        finally:
            GoObj.set("allowed_evidence_codes", None)

    def test_set_allowed_evidence_codes_raises_GeneOntologyError(self, GoObj):
        with raises(GeneOntologyError):
            GoObj.set_allowed_evidence_codes(["NOTAKEY"])

    def test_set(self, GoObj):
        name = "TESTNAME"
        value = "TESTVALUE"
        GoObj.set(name, value)
        assert hasattr(GoObj, name)
        assert getattr(GoObj, name) == value

    def test_get(self, GoObj):
        name = "TESTNAME"
        value = "TESTVALUE"
        setattr(GoObj, name, value)
        assert hasattr(GoObj, name)
        assert GoObj.get(name) == value

    # @mark.io
    def test_load_go_dag(self, GoObj):
        assert hasattr(GoObj, "go_dag")

    def test_load_go_dag_raises_GeneOntologyError(self, GoObj):
        with raises(GeneOntologyError):
            GoObj.load_go_dag("NOTAPATH")

    def test_iter(self, GoObj):
        assert inspect.isgenerator(iter(GoObj))

    def test_iter_raises_GeneOntologyError(self, GoObj):
        try:
            GoObj.set("custom_association_file_path", "NOTAPATH")
            with raises(GeneOntologyError):
                iterator = iter(GoObj)
                next(iterator)
        finally:
            delattr(GoObj, "custom_association_file_path")

    def test_iter_next_no_evidence_codes(self, GoObj):
        iterator = iter(GoObj)
        assert isinstance(next(iterator), dict)

    def test_iter_next_with_evidence_codes(self, GoObj):
        try:
            GoObj.set_allowed_evidence_codes(self.example_evidence_codes)
            iterator = iter(GoObj)
            assert isinstance(next(iterator), dict)
        finally:
            GoObj.set("allowed_evidence_codes", None)

    def test_remove_unwanted_genes(self):
        d = {"a": 1, "b": 2, "c": 3}
        w = set(("c"))
        assert self.obj.remove_unwanted_genes(w, d) == {"c": 3}

    def test_term2ontology(self, GoObj):
        assert isinstance(GoObj.term2ontology(), dict)

    def test_ontology2term(self, GoObj):
        assert isinstance(GoObj.ontology2term(), dict)

    def test_get_associations(self, GoObj):
        d = GoObj.get_associations()
        assert isinstance(d, defaultdict)
        assert hasattr(GoObj, "all_associations")
        assert isinstance(GoObj.all_associations, defaultdict)
        assert hasattr(GoObj, "associations")
        assert len(list(itertools.chain(*d.values()))) == 37954

    def test_get_associations_raises_GeneOntologyError(self, GoObj):
        with raises(GeneOntologyError):
            GoObj.get_associations(ontology="NOTAVALUE")

    def test_get_associations_ontology_P(self, GoObj):
        d = GoObj.get_associations(ontology="P")
        assert len(list(itertools.chain(*d.values()))) == 12773

    def test_get_associations_ontology_F(self, GoObj):
        d = GoObj.get_associations(ontology="F")
        assert len(list(itertools.chain(*d.values()))) == 9498

    def test_get_associations_ontology_C(self, GoObj):
        d = GoObj.get_associations(ontology="C")
        assert len(list(itertools.chain(*d.values()))) == 15711


class TestPrettify(object):
    def test_prettify(self):
        assert prettify("biological_process") == "Biological process"

    def test_raises_GeneOntologyError(self):
        with raises(GeneOntologyError):
            prettify("NOTVALID")
