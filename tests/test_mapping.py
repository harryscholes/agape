from pytest import raises
import pandas as pd

from agape.mapping import dictify, gene2symbol


class TestDictify(object):
    def setup_method(self):
        self.df = pd.DataFrame({"A": [0, 1],
                                "B": ["x", "y"]})
        self.key = "A"
        self.value = "B"

    def test_returns_dict(self):
        assert isinstance(dictify(self.df, self.key, self.value), dict)

    def test_df_equals_dict(self):
        d = dictify(self.df, self.key, self.value)
        assert all(self.df[self.key].values == list(d.keys()))
        assert all(self.df[self.value].values == list(d.values()))

    def test_raises_keyerror(self):
        with raises(KeyError):
            dictify(self.df, self.key, "C")


class TestGene2Symbol(object):
    def setup_method(self):
        self.d = gene2symbol("ID", "Symbol")

    def test_returns_dict(self):
        assert isinstance(self.d, dict)

    def test_raises_keyerror(self):
        with raises(KeyError):
            gene2symbol("ID", "NOTAKEY")
