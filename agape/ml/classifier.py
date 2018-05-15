'''Classifiers.
'''
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

__all__ = ["SVClassifier", "RFClassifier"]


class Classifier:
    '''Classifier base class.
    '''
    def __init__(self, clf, scale=False):
        pipeline = []

        if scale is True:
            pipeline.append(('standard_scaler', StandardScaler()))

        pipeline.append(('estimator', clf))
        self.clf = Pipeline(pipeline)

    def __name__(self):
        return self.__class__.__name__

    def grid_search(self, X, y, parameters):
        self.grid_search_parameters = {'estimator__' + k: v for k, v
                                       in parameters.items()}
        clf = self.clf
        self.clf_grid_search = GridSearchCV(clf, self.grid_search_parameters)
        self.clf_grid_search.fit(X, y)
        print('`clf.best_estimator_`:\n\n',
              self.clf_grid_search.best_estimator_, sep='')

    def fit(self, X, y):
        # Fit classifier using the best parameters from GridSearchCV
        try:
            getattr(self.clf_grid_search, 'best_estimator_')
            fit_using = "clf_grid_search"
        except AttributeError:
            # Fit classifier from __init__
            fit_using = "clf"
            self.clf.fit(X, y)
        finally:
            print(f'\nFit using `{fit_using}`')

    def get_clf(self):
        try:
            return self.clf_grid_search.best_estimator_
        except AttributeError:
            return self.clf

    def predict(self, X):
        self.predictions = self.get_clf().predict(X)
        return self.predictions

    def predict_proba(self, X):
        self.probabilities = self.get_clf().predict_proba(X)
        return self.probabilities

    def accuracy(self, y):
        self.accuracy_score = accuracy_score(y, self.predictions)
        return self.accuracy_score


class SVClassifier(Classifier):
    '''Support Vector Classifier.
    '''
    def __init__(self, random_state=None):
        super().__init__(
            OneVsRestClassifier(
                SVC(probability=True,
                    random_state=random_state),
                n_jobs=-1),
            scale=True)


class RFClassifier(Classifier):
    '''Random Forest Classifer.
    '''
    def __init__(self, random_state=None):
        super().__init__(
            OneVsRestClassifier(
                RandomForestClassifier(
                    n_jobs=-1,
                    random_state=random_state),
                n_jobs=-1),
            scale=True)
