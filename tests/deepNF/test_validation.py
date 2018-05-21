from pytest import raises
import numpy as np
from agape.deepNF.validation import m_AUPR, M_AUPR


class Test_m_AUPR:
    def test_m_AUPR(self):
        np.random.seed(0)
        assert np.allclose(m_AUPR(
            np.random.randint(0, 2, 10),
            np.random.rand(10)), 0.64379960317460305)

        np.random.seed(1)
        assert np.allclose(m_AUPR(
            np.random.randint(0, 2, 10),
            np.random.rand(10)), 0.74092970521541957)


class Test_M_AUPR:
    def test_M_AUPR(self):
        np.random.seed(0)
        assert np.allclose(M_AUPR(
            np.random.randint(0, 2, (10, 3)),
            np.random.rand(10, 3)), 0.50949257789535574)

        np.random.seed(1)
        assert np.allclose(M_AUPR(
            np.random.randint(0, 2, (10, 3)),
            np.random.rand(10, 3)), 0.4966402116402116)

    def test_raises_ValueError(self):
        with raises(ValueError):
            A = np.arange(5)
            M_AUPR(A, A)
