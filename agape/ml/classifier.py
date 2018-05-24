'''Classifiers.
'''
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from ..base import Base

__all__ = ["SVClassifier", "RFClassifier"]


class Classifier(Base):
    '''Classifier base class.
    '''
    def __init__(self, clf, scale=False, n_jobs=1):
        super().__init__()

        # A classifier is built using a `Pipeline` for convenience of chaining
        # multiple preprocessing steps before the classifier
        pipeline = []
        # Centre data by scaling to zero mean and unit variance
        if scale is True:
            pipeline.append(('standard_scaler', StandardScaler()))
        # Add the `clf` estimator and build the `Pipeline`
        pipeline.append(('estimator', clf))
        self.clf = Pipeline(pipeline)
        self.n_jobs = n_jobs

    def __name__(self):
        return self.__class__.__name__

    def grid_search(self, X, y, parameters, scoring=None, cv=5, refit=True,
                    verbose=False):
        '''Perform an exhaustive search over hyperparameter combinations.

        # Arguments
            X: np.ndarray, features
            y: np.ndarray, labels
            parameters: dict, hyperparameter ranges
            scoring: dict, scoring functions e.g. {'acc': accuracy_score, ...}
            refit: bool, fit an estimator with the best parameters if True
            verbose: int, controls the verbosity: the higher, the more messages
        '''
        self.grid_search_parameters = {'estimator__' + k: v for k, v
                                       in parameters.items()}
        clf = self.clf

        self.clf_grid_search = GridSearchCV(
            clf,
            self.grid_search_parameters,
            cv=cv,
            scoring=scoring,
            refit=refit,
            n_jobs=self.n_jobs,
            verbose=verbose)

        self.clf_grid_search.fit(X, y)
        print('\n`clf.best_estimator_`:\n',
              self.clf_grid_search.best_estimator_, '\n', sep='')

    def fit(self, X, y):
        '''Fit the estimator.

        # Arguments
            X: np.ndarray, features
            y: np.ndarray, labels
        '''
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
        '''Get the best estimator.

        If a grid search has been performed, then the `best_estimator_` is
        returned, else the estimator used to initialise the object is returned.

        # Returns
            clf: sklearn estimator
        '''
        try:
            return self.clf_grid_search.best_estimator_
        except AttributeError:
            return self.clf

    def predict(self, X):
        '''Predict the classes of samples using features.

        # Arguments
            X: np.ndarray, features

        # Returns
            predictions: np.ndarray, class predictions
        '''
        self.predictions = self.get_clf().predict(X)
        return self.predictions

    def predict_proba(self, X):
        '''Predict the class-membership probabilities of samples.

        # Arguments
            X: np.ndarray, features

        # Returns
            probabilities: np.ndarray, class probabilities
        '''
        self.probabilities = self.get_clf().predict_proba(X)
        return self.probabilities

    def decision_function(self, X):
        '''Decision function.

        # Arguments
            X: np.ndarray, features

        # Returns
            decisions: np.ndarray, distances of samples to the decision
                boundary
        '''
        try:
            self.decisions = self.get_clf().decision_function(X)
            return self.decisions
        except AttributeError as err:
            raise AttributeError(
                f'decision_function is not implemented for {self.__name__()}')\
                from None

    def accuracy(self, y_true):
        '''Accuracy score.

        # Arguments
            y_true: np.ndarry, true labels

        # Returns
            accuracy_score: float
        '''
        self.accuracy_score = accuracy_score(y_true, self.predictions)
        return self.accuracy_score


class SVClassifier(Classifier):
    '''Support Vector Classifier.
    '''
    def __init__(self, random_state=None, n_jobs=1):
        super().__init__(
            OneVsRestClassifier(
                SVC(probability=True,
                    random_state=random_state),
                n_jobs=n_jobs),
            scale=True)


class RFClassifier(Classifier):
    '''Random Forest Classifer.
    '''
    def __init__(self, random_state=None, n_jobs=1):
        super().__init__(
            OneVsRestClassifier(
                RandomForestClassifier(
                    n_jobs=n_jobs,
                    random_state=random_state),
                n_jobs=n_jobs))
