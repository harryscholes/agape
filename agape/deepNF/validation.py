import numpy as np
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import ShuffleSplit
from scipy.stats import sem as std
from agape.ml.classifier import SVClassifier
from agape.utils import stdout
from collections import defaultdict


def AUPR(y_true, y_score):
    '''Area under the precision-recall curve.
    '''
    y_true = y_true.flatten()
    y_score = y_score.flatten()
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    P = np.count_nonzero(y_true)
    TP = np.cumsum(y_true, dtype=float)
    PP = np.arange(1, len(y_true)+1, dtype=float)
    recall = np.divide(TP, P)
    precision = np.divide(TP, PP)
    pr = np.trapz(precision, recall)
    return pr


def M_AUPR(y_true, y_score):
    '''Macro-average AUPR.

    Computes AUPR independently for each class and returns the mean.
    '''
    AUC = 0.0
    n = 0
    for i in range(y_true.shape[1]):
        AUC_i = AUPR(y_true[:, i], y_score[:, i])
        if sum(y_true[:, i]) > 0:
            n += 1
            AUC += AUC_i
    AUC /= n
    return AUC


def m_AUPR(y_true, y_score):
    '''Micro-average AUPR.

    Computes AUPR across all classes.
    '''
    AUC = AUPR(y_true, y_score)
    return AUC


def evaluate_performance(y_true, y_score, y_pred) -> dict:
    '''Evaluate performance.
    '''
    perf = {}
    # Compute macro-averaged AUPR
    perf['M_AUPR'] = M_AUPR(y_true, y_score)
    # Compute micro-averaged AUPR
    perf['m_AUPR'] = m_AUPR(y_true, y_score)
    # Computes accuracy
    perf['accuracy'] = accuracy_score(y_true, y_pred)
    # Compute F1-score
    perf["f1"] = f1_score(y_true, y_pred, average='micro')
    return perf


def cross_validation(X, y, n_trials=10, n_jobs=1,
                     random_state=None):
    '''Perform model selection via cross validation.
    '''
    stdout('Number of samples pre-filtering', X.shape)

    # Filter samples with no annotations
    del_rid = np.where(y.sum(axis=1) == 0)[0]
    y = np.delete(y, del_rid, axis=0)
    X = np.delete(X, del_rid, axis=0)
    stdout('Number of samples post-filtering', X.shape)

    # Hyperparameters
    C = np.logspace(0, 1, 2)
    gamma = np.logspace(-1, -0, 2)

    grid_search_params = {
        'estimator__C': C,
        'estimator__gamma': gamma,
        'estimator__kernel': ['rbf']}

    # Scoring
    scoring = {
        'accuracy': 'accuracy',
        'f1': 'f1_micro',
        'M_AUPR': make_scorer(M_AUPR),
        'm_AUPR': make_scorer(m_AUPR)}

    # Classifier
    clf = SVClassifier(n_jobs=n_jobs, random_state=random_state)

    # Split training data
    trials = ShuffleSplit(n_splits=n_trials,
                          test_size=0.2,
                          random_state=random_state)

    # Performance
    performance_metrics = ("accuracy", "m_AUPR", "M_AUPR", "f1")
    perf = defaultdict(dict)
    perf['grid_search'] = {}

    # Model selection for optimum hyperparameters
    iteration = 0
    for train_idx, test_idx in trials.split(X):
        # Split data
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        iteration += 1
        stdout('Cross validation trial', iteration)
        stdout('Train samples', y_train.shape[0])
        stdout('Test samples', y_test.shape[0])

        # Perform a grid search over the hyperparameter ranges

        stdout('Grid search')

        clf.grid_search(
            X_train,
            y_train,
            grid_search_params,
            scoring=scoring,
            refit='m_AUPR',
            cv=2)  # TODO temp

        # Get the best hyperparameters
        clf_params = clf.get_clf().get_params()['estimator'] \
                                  .get_params()['estimator'] \
                                  .get_params()
        best_params = {k: clf_params[k.replace('estimator__', '')]
                       for k in grid_search_params}

        perf['grid_search'][iteration] = {}
        perf['grid_search'][iteration]['best_estimator_'] = {}

        for k, v in best_params.items():
            k = k.split('__')[-1]
            perf['grid_search'][iteration]['best_estimator_'][k] = v

        stdout('Optimal parameters', best_params)

        perf['grid_search'][iteration]['best_score_'] = \
            clf.clf_grid_search.best_score_

        stdout('Train dataset AUPR', clf.clf_grid_search.best_score_)


        # Train a classifier with the optimal hyperparameters using the full
        # training data
        clf.fit(X_train, y_train)

        # Compute performance on test set
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)

        stdout('Number of positive predictions', len(y_pred.nonzero()[0]))

        perf_trial = evaluate_performance(y_test, y_score, y_pred)

        for pm in performance_metrics:
            perf[pm][iteration] = perf_trial[pm]

        stdout('Test dataset')

        for measure, value in perf_trial.items():
            if not isinstance(measure, int):
                stdout(measure, value)

    # Performance across K-fold cross-validation

    def calculate_mean_std(metric):
        values = list(perf[metric].values())
        perf['metrics'][metric]['mean'] = np.mean(values)
        perf['metrics'][metric]['std'] = std(values)

    for pm in performance_metrics:
        perf['metrics'][pm] = {}
        calculate_mean_std(pm)

    return perf
