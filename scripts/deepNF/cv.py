'''
Train a classifier on node embeddings with k-fold cross-validation.

Usage:
    python cv.py
'''
import os
import argparse
import json
from pprint import pprint
import scipy.io as sio
from agape.deepNF.validation import cross_validation
from agape.deepNF.utils import load_embeddings, mkdir
from agape.utils import stdout, directory_exists
import sklearn
import warnings
import glob

print(__doc__)

##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--level', required=True)
parser.add_argument('-o', '--organism', default='yeast', type=str)
parser.add_argument('-t', '--model-type', default='mda', type=str)
parser.add_argument('-m', '--models-path', default="models", type=str)
parser.add_argument('-r', '--results-path', default="results", type=str)
parser.add_argument('-d', '--data-path', default="$AGAPEDATA/deepNF", type=str)
parser.add_argument('-n', '--n-trials', default=5, type=int)
parser.add_argument('-v', '--validation', default='cv', type=str)
parser.add_argument('-s', '--random_state', default=-1, type=int,
                    help='Set sklearn random_state. If -1, then sklearn uses \
                          the system randomness as a seed. If int, then this \
                          number will be used as a seed.')
parser.add_argument('--tags', default="", type=str)
parser.add_argument('-j', '--n_jobs', default=1, type=int)
parser.add_argument('-c', '--clf_type', default='LRC', type=str)
parser.add_argument('--test', default=None, type=int,
                    help='If True, then only a subset of the data is used')
args = parser.parse_args()

stdout("Command line arguments", args)

org = args.organism
model_type = args.model_type
models_path = os.path.expandvars(args.models_path)
results_path = os.path.expandvars(args.results_path)
data_path = os.path.expandvars(args.data_path)
n_trials = args.n_trials
tags = args.tags
validation = args.validation
n_jobs = args.n_jobs
random_state = args.random_state
level = args.level
clf_type = args.clf_type
test = args.test


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
    'cerevisiae': ('level1', 'level2', 'level3')}

try:
    annotation_types = validation_types[validation]
except KeyError as err:
    err.args = (f'Not a valid validation type: {validation}',)

if level not in annotation_types:
    raise ValueError(f'Level must be one of:', annotation_types)


# Performance measures
measures = ('m-aupr_avg', 'm-aupr_std', 'M-aupr_avg', 'M-aupr_std',
            'F1_avg', 'F1_std', 'acc_avg', 'acc_std')


########
# defs #
########

def main():
    ######################
    # Prepare filesystem #
    ######################

    directory_exists(models_path)
    mkdir(results_path)

    ###################
    # Load embeddings #
    ###################

    embeddings_file = glob.glob(os.path.join(models_path, '*.mat'))[0]
    model_name = os.path.basename(embeddings_file).split('.')[0]
    print(model_name)
    stdout('Loading embeddings', embeddings_file)
    embeddings = load_embeddings(embeddings_file)

    #######################
    # Load GO annotations #
    #######################

    annotation_dir = os.path.join(data_path, 'annotations')
    if validation == 'cerevisiae':
        annotation_file = os.path.join(annotation_dir, 'cerevisiae_annotations.mat')
    else:
        annotation_file = os.path.join(annotation_dir, 'yeast_annotations.mat')
    stdout('Loading GO annotations', annotation_file)

    GO = sio.loadmat(annotation_file)

    ####################
    # Train classifier #
    ####################

    stdout('Running cross-validation for', level)

    annotations = GO[level]

    # Silence certain warning messages during cross-validation
    for w in (sklearn.exceptions.UndefinedMetricWarning, UserWarning,
              RuntimeWarning):
        warnings.filterwarnings("ignore", category=w)

    # Only use a subset of the data for testing purposes
    embeddings = embeddings[:test]
    annotations = annotations[:test]

    performance = cross_validation(
        embeddings,
        annotations,
        n_trials=n_trials,
        n_jobs=n_jobs,
        random_state=random_state,
        clf_type=clf_type)

    performance['level'] = level

    pprint(performance)

    fout = f'{model_name}_{level}_{clf_type}_{validation}_performance.json'

    with open(os.path.join(results_path, fout), 'w') as f:
        json.dump(performance, f)


if __name__ == '__main__':
    main()
