from pytest import fixture
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from agape.ml.classifier import Classifier, SVClassifier, RFClassifier


random_state = 0
X, y = make_classification(n_samples=10, n_features=5,
                           random_state=random_state)


@fixture(scope='module')
def Clf():
    '''Classifier fit using `fit`.
    '''
    clf = Classifier(
        OneVsRestClassifier(
            LogisticRegression(random_state=random_state),
            n_jobs=-1),
        scale=True)
    clf.fit(X, y)
    return clf


@fixture(scope='module')
def ClfGS():
    '''Classifier fit using `grid_search`.
    '''
    clf = Classifier(
        OneVsRestClassifier(
            LogisticRegression(random_state=random_state),
            n_jobs=-1),
        scale=True)
    parameters = {'estimator__C': np.logspace(-1, 1, 3)}
    clf.grid_search(X, y, parameters)
    return clf


class TestClassifier:
    def test_predict(self, Clf):
        assert np.array_equal(Clf.predict(X), [0, 0, 1, 0, 1, 1, 0, 1, 1, 0])

    def test_predict_proba(self, Clf):
        expected = np.array(
            [[0.86175027, 0.13824973],
             [0.92458054, 0.07541946],
             [0.02817212, 0.97182788],
             [0.83849173, 0.16150827],
             [0.2650148, 0.7349852],
             [0.25549562, 0.74450438],
             [0.75834918, 0.24165082],
             [0.0713748, 0.9286252],
             [0.40150536, 0.59849464],
             [0.67087362, 0.32912638]])
        assert np.allclose(Clf.predict_proba(X), expected)

    def test_accuracy(self, Clf):
        Clf.predict(X)
        assert Clf.accuracy(y) == 1.0

        Clf.predictions = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1])
        assert np.allclose(Clf.accuracy(y), 0.699999)

    def test_get_clf_Clf(self, Clf):
        assert hasattr(Clf, 'clf')
        assert not hasattr(Clf, 'clf_grid_search')
        clf = Clf.get_clf()
        assert isinstance(clf, Pipeline)

    def test_get_clf_ClfGS(self, ClfGS):
        assert hasattr(ClfGS, 'clf_grid_search')
        clf = ClfGS.get_clf()
        assert isinstance(clf, Pipeline)


@fixture(scope='module')
def SVClf():
    '''SVClassifier instance.
    '''
    clf = SVClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf


class TestSVClassifier:
    def test_predict(self, SVClf):
        assert np.array_equal(SVClf.predict(X), [0, 0, 1, 0, 1, 1, 0, 1, 1, 0])

    def test_predict_proba(self, SVClf):
        expected = np.array(
            [[0.96709295, 0.03290705],
             [0.96708461, 0.03291539],
             [0.00886844, 0.99113156],
             [0.95934577, 0.04065423],
             [0.00885798, 0.99114202],
             [0.00885607, 0.99114393],
             [0.92929739, 0.07070261],
             [0.00885798, 0.99114202],
             [0.01053052, 0.98946948],
             [0.76418338, 0.23581662]])
        print(SVClf.predict_proba(X))
        assert np.allclose(SVClf.predict_proba(X), expected)


@fixture(scope='module')
def RFClf():
    '''RFClassifier instance.
    '''
    clf = RFClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf


class TestRFClassifier:
    def test_predict(self, RFClf):
        assert np.array_equal(RFClf.predict(X), [0, 0, 1, 0, 1, 1, 0, 1, 1, 0])

    def test_predict_proba(self, RFClf):
        expected = np.array(
            [[1.0, 0.0],
             [0.9, 0.1],
             [0.1, 0.9],
             [0.8, 0.2],
             [0.1, 0.9],
             [0.2, 0.8],
             [0.8, 0.2],
             [0.2, 0.8],
             [0.2, 0.8],
             [0.8, 0.2]])
        print(RFClf.predict_proba(X))
        assert np.allclose(RFClf.predict_proba(X), expected)
