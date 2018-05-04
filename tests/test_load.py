from pytest import raises, fixture
import pandas as pd
from agape.load import Genes, Biogrid, STRING


@fixture(scope="module")
def GenesObj():
    """Instantiate `Genes` object.
    """
    genes = Genes()
    return genes

class TestGenes:
    def setup_method(self):
        self.obj = Genes()
        self.obj.viability()

    def test_has_attrs(self, GenesObj):
        assert hasattr(GenesObj, "df")
        assert hasattr(GenesObj, "genes")

    def test_attrs_types(self, GenesObj):
        assert isinstance(GenesObj.df, pd.DataFrame)
        assert isinstance(GenesObj.genes, list)

    def test_viability(self, GenesObj):
        df = GenesObj.viability()
        assert df.shape == (4917, 2)

    def test_viability_viable(self, GenesObj):
        genes = GenesObj.viability("viable")
        assert isinstance(genes, list)
        assert len(genes) == 3570

    def test_viability_inviable(self, GenesObj):
        genes = GenesObj.viability("inviable")
        assert isinstance(genes, list)
        assert len(genes) == 1243

    def test_viability_viable(self, GenesObj):
        genes = GenesObj.viability("condition-dependent")
        assert isinstance(genes, list)
        assert len(genes) == 104

    def test_viability_raises_KeyError(self, GenesObj):
        with raises(KeyError):
            GenesObj.viability("NOTAKEY")


@fixture(scope="module")
def BiogridObj():
    return Biogrid()


class TestBiogrid:
    def test_load_biogrid(self, BiogridObj):
        assert BiogridObj().shape == (71990, 24)

    def test_load_biogrid_physical(self, BiogridObj):
        assert BiogridObj(interaction_type="physical").shape == (12994, 24)

    def test_load_biogrid_genetic(self, BiogridObj):
        assert BiogridObj(interaction_type="genetic").shape == (58996, 24)

    def test_load_graph(self, BiogridObj):
        assert BiogridObj(graph=True).shape == (71990, 2)

    def test_raises_KeyError(self, BiogridObj):
        with raises(KeyError):
            BiogridObj(interaction_type="NOTAKEY")


@fixture(scope="module")
def STRINGObj():
    return STRING()


class TestSTRING:
    def test_string_interaction_types(self, STRINGObj):
        assert hasattr(STRINGObj, "interaction_types")
        assert isinstance(STRINGObj.interaction_types, tuple)
        assert STRINGObj.interaction_types == (
            'neighborhood', 'fusion', 'cooccurence', 'coexpression',
            'experimental', 'database')

    def test_string_df(self, STRINGObj):
        assert hasattr(STRINGObj, "df")
        assert isinstance(STRINGObj.df, pd.DataFrame)
        assert STRINGObj.df.shape == (1657582, 10)

    def test_get(self, STRINGObj):
        df = STRINGObj.get("experimental")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1657582, 3)

    def test_get_raises_KeyError(self, STRINGObj):
        with raises(KeyError):
            STRINGObj.get(interaction_type="NOTAKEY")
