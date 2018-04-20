from pytest import raises
import pandas as pd
from agape.load import Genes, biogrid


class TestGenes(object):
    def setup_method(self):
        self.obj = Genes()
        self.obj.viability()

    def test_has_attrs(self):
        assert hasattr(self.obj, "df")
        assert hasattr(self.obj, "genes")

    def test_attrs_types(self):
        assert isinstance(self.obj.df, pd.DataFrame)
        assert isinstance(self.obj.genes, list)

    def test_viability(self):
        df = self.obj.viability()
        assert df.shape == (4917, 2)

    def test_viability_viable(self):
        genes = self.obj.viability("viable")
        assert isinstance(genes, list)
        assert len(genes) == 3570

    def test_viability_inviable(self):
        genes = self.obj.viability("inviable")
        assert isinstance(genes, list)
        assert len(genes) == 1243

    def test_viability_viable(self):
        genes = self.obj.viability("condition-dependent")
        assert isinstance(genes, list)
        assert len(genes) == 104

    def test_viability_raises_KeyError(self):
        with raises(KeyError):
            self.obj.viability("NOTAKEY")


class TestBiogrid(object):
    def test_load_biogrid(self):
        df = biogrid()
        assert df.shape == (71990, 24)

    def test_load_biogrid_physical(self):
        df = biogrid(interaction_type="physical")
        assert df.shape == (12994, 24)

    def test_load_biogrid_genetic(self):
        df = biogrid(interaction_type="genetic")
        assert df.shape == (58996, 24)

    def test_load_graph(self):
        df = biogrid(graph=True)
        assert df.shape == (71990, 2)

    def test_raises_KeyError(self):
        with raises(KeyError):
            biogrid(interaction_type="NOTAKEY")
