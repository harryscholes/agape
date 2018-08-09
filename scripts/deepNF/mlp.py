'''
Train an MLP on node embeddings with k-fold cross-validation.

Usage:
    python mlp.py
'''
import os
import json
import argparse
import warnings
import glob
from pprint import pprint
from collections import defaultdict
import numpy as np
import scipy.io as sio
from scipy.stats import sem as std
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from agape.deepNF.utils import load_embeddings, mkdir
from agape.utils import stdout, directory_exists
from agape.deepNF.validation import _Performance

print(__doc__)


# NOTE Define the architecture of the MLP model here

def MLP(x, y):
    '''MLP architecture.
    '''
    model = Sequential([
        Dense(512, activation='relu', input_shape=(x.shape[1],)),
        # BatchNormalization(),
        Dropout(.5),
        Dense(256, activation='relu'),
        # BatchNormalization(),
        Dropout(.5),
        Dense(y.shape[1], activation='sigmoid')])

    return model


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--level', required=True)
parser.add_argument('-o', '--organism', default='yeast', type=str)
parser.add_argument('-v', '--validation', default='cv', type=str)
parser.add_argument('-m', '--models-path', default="models", type=str)
parser.add_argument('-r', '--results-path', default="results", type=str)
parser.add_argument('-d', '--data-path', default="$AGAPEDATA/deepNF", type=str)
parser.add_argument('-n', '--n-trials', default=10, type=int)
parser.add_argument('-c', '--clf-type', required=True, type=str)
parser.add_argument('-s', '--random-state', default=-1, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--tags', default="", type=str)
args = parser.parse_args()

stdout("Command line arguments", args)

level = args.level
org = args.organism
validation = args.validation
models_path = os.path.expandvars(args.models_path)
results_path = os.path.expandvars(args.results_path)
data_path = os.path.expandvars(args.data_path)
n_trials = args.n_trials
random_state = args.random_state
tags = args.tags
clf_type = args.clf_type
batch_size = args.batch_size

# Set random_state seed for sklearn
if random_state == -1:
    random_state = None  # Seed randomness with system randomness
elif random_state > 0:
    pass  # Seed randomness with random_state
else:
    raise ValueError('--random_state must be -1 or > 0')

# Validation type
validation_types = {
    'cv': ('P_1', 'P_2', 'P_3', 'F_1', 'F_2', 'F_3', 'C_1', 'C_2', 'C_3'),
    'cerevisiae': ('level1', 'level2', 'level3', 'all')}

try:
    annotation_types = validation_types[validation]
except KeyError as err:
    err.args = (f'Not a valid validation type: {validation}',)

if level not in annotation_types:
    raise ValueError(f'Level must be one of:', annotation_types)


def main():
    # Prepare filesystem
    directory_exists(models_path)
    mkdir(results_path)

    # Load embeddings
    embeddings_file = glob.glob(os.path.join(models_path, '*.mat'))[0]
    model_name = os.path.splitext(
        os.path.basename(embeddings_file))[0].replace('_embeddings', '')
    stdout('Loading embeddings', embeddings_file)
    embeddings = load_embeddings(embeddings_file).astype('int32')

    # Load annotations
    annotation_dir = os.path.join(data_path, 'annotations')
    if validation == 'cerevisiae':
        annotation_file = os.path.join(
            annotation_dir, 'cerevisiae_annotations.mat')
    else:
        annotation_file = os.path.join(annotation_dir, 'yeast_annotations.mat')
    stdout('Loading GO annotations', annotation_file)

    annotation_file = sio.loadmat(annotation_file)

    # Train classifier
    stdout('Running cross-validation for', level)

    if level == 'all':
        annotations = np.hstack((annotation_file['level1'],
                                 annotation_file['level2'],
                                 annotation_file['level3'])).astype('int32')
    else:
        annotations = annotation_file[level].astype('int32')

    # Silence certain warning messages during cross-validation
    for w in (sklearn.exceptions.UndefinedMetricWarning, UserWarning,
              RuntimeWarning):
        warnings.filterwarnings("ignore", category=w)

    # Remove genes with no annotations
    x = embeddings
    y = annotations
    del_rid = np.where(y.sum(axis=1) == 0)[0]
    x = np.delete(x, del_rid, axis=0)
    y = np.delete(y, del_rid, axis=0)

    # Set up CV
    performance_metrics = (
        "accuracy", "m_AUPR", "M_AUPR", "f1", "precision", "recall")
    performance = defaultdict(dict)

    trials = ShuffleSplit(n_splits=n_trials, test_size=0.2,
                          random_state=random_state)
    iteration = 0

    # CV-folds
    for train_idx, test_idx in trials.split(x):
        iteration += 1

        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Define the MLP architecture
        model = MLP(x_train, y_train)
        model.compile('adam', 'binary_crossentropy', ['acc'])

        # Train the model
        callbacks = [EarlyStopping(min_delta=0., patience=20),
                     ModelCheckpoint('best_model.h5', save_best_only=True)]

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=200,
                            validation_split=0.2, shuffle=True,
                            callbacks=callbacks, verbose=2)

        performance['history'][iteration] = {}
        for tm in history.history:
            performance['history'][iteration][tm] = history.history[tm]

        # Read the best model from file (defined as the model which minimizes
        # the validation loss.
        model = load_model('best_model.h5')

        # Predict annotations
        y_score = model.predict(x_test)
        y_pred = y_score.copy()
        positive_threshold = .5
        y_pred[y_pred < positive_threshold] = 0
        y_pred[y_pred > 0] = 1
        performance_trial = _Performance(y_test, y_score, y_pred)

        for pm in performance_metrics:
            performance[pm][iteration] = getattr(performance_trial, pm)
            calculate_mean_std(performance, pm)

        dummy = DummyClassifier().fit(x_train, y_train).score(x_test, y_test)
        performance['dummy'][iteration] = dummy

    performance['level'] = level
    pprint(performance)

    # Save results and training history
    fout = f'{model_name}_{level}_{clf_type}'
    with open(os.path.join(results_path, f'{fout}.json'), 'w') as f:
        json.dump(performance, f)

    # Delete the best model file
    os.remove('best_model.h5')

    return None


def calculate_mean_std(performance, metric):
    performance['metrics'][metric] = {}
    values = list(performance[metric].values())
    performance['metrics'][metric]['mean'] = np.mean(values)
    performance['metrics'][metric]['std'] = std(values)


if __name__ == '__main__':
    main()
