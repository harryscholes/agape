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
# import sklearn
# import warnings

# warnings.filterwarnings(
#     "ignore",
#     category=sklearn.exceptions.UndefinedMetricWarning)

print(__doc__)

##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gene-ontology', type=str, required=True)
parser.add_argument('-l', '--level', type=int, required=True)
parser.add_argument('-o', '--organism', default='yeast', type=str)
parser.add_argument('-t', '--model-type', default='mda', type=str)
parser.add_argument('-m', '--models-path', default="models", type=str)
parser.add_argument('-r', '--results-path', default="results", type=str)
parser.add_argument('-d', '--data-path', default="$AGAPEDATA/deepNF", type=str)
parser.add_argument('-a', '--architecture', default=2, type=int)
parser.add_argument('-n', '--n-trials', default=5, type=int)
parser.add_argument('-v', '--validation', default='cv', type=str)
parser.add_argument('-s', '--random_state', default=-1, type=int,
                    help='Set sklearn random_state. If -1, then sklearn uses \
                          the system randomness as a seed. If int, then this \
                          number will be used as a seed.')
parser.add_argument('--tags', default="", type=str)
parser.add_argument('-j', '--n_jobs', default=1, type=int)
args = parser.parse_args()

stdout("Command line arguments", args)

org = args.organism
model_type = args.model_type
models_path = os.path.expandvars(args.models_path)
results_path = os.path.expandvars(args.results_path)
data_path = os.path.expandvars(args.data_path)
architecture = args.architecture
n_trials = args.n_trials
tags = args.tags
validation = args.validation
n_jobs = args.n_jobs
random_state = args.random_state
level = args.level
gene_ontology = args.gene_ontology


# Set random_state seed for sklearn
if random_state == -1:
    random_state = None  # Seed randomness with system randomness
elif random_state > 0:
    pass  # Seed randomness with random_state
else:
    raise ValueError('--random_state must be -1 or > 0')


# Validation type
validation_types = {
    'cv': ('P_3', 'P_2', 'P_1', 'F_3', 'F_2', 'F_1', 'C_3', 'C_2', 'C_1')}

try:
    annotation_types = validation_types[validation]
except KeyError as err:
    err.args = (f'Not a valid validation type: {validation}',)


# Ontology
if gene_ontology not in {'P', 'F', 'C'}:
    raise ValueError('--gene-ontology must be P, F or C')


# Level
if level not in {1, 2, 3}:
    raise ValueError('--level must be 1, 2 or 3')

level = f'{gene_ontology}_{level}'


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

    model_name = f'{org}_{model_type.upper()}_arch_{architecture}'
    embeddings_file = f'{model_name}_features.mat'
    embeddings_path = os.path.join(models_path, embeddings_file)
    stdout('Loading embeddings', embeddings_path)
    embeddings = load_embeddings(embeddings_path)

    #######################
    # Load GO annotations #
    #######################

    if validation == 'cv':
        annotation_dir = os.path.join(
            os.path.expandvars('$AGAPEDATA'), 'annotations')
        annotation_file = 'yeast_annotations.mat'
        stdout('Loading GO annotations')

        GO = sio.loadmat(
            os.path.join(annotation_dir, annotation_file))

    ####################
    # Train classifier #
    ####################

    if validation == 'cv':
        stdout('Running cross-validation for', level)

        performance = cross_validation(
            embeddings,
            GO[level],
            n_trials=n_trials,
            n_jobs=n_jobs,
            random_state=random_state)

        pprint(performance)

        fout = f'{model_name}_{level}_{validation}_performance.json'

        with open(os.path.join(results_path, fout), 'w') as f:
            json.dump(performance, f)


if __name__ == '__main__':
    main()
