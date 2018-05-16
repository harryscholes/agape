'''
Train a classifier on node embeddings with k-fold cross-validation.

Usage:
    python cv.py
'''
import os
import argparse
import scipy.io as sio
from agape.deepNF.validation import cross_validation
from agape.deepNF.utils import load_embeddings, mkdir
from agape.utils import stdout, directory_exists
import sklearn
import warnings

warnings.filterwarnings(
    "ignore",
    category=sklearn.exceptions.UndefinedMetricWarning)

print(__doc__)

##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--organism', default='yeast', type=str)
parser.add_argument('-t', '--model-type', default='mda', type=str)
parser.add_argument('-m', '--models-path', default="models", type=str)
parser.add_argument('-r', '--results-path', default="results", type=str)
parser.add_argument('-d', '--data-path', default="$AGAPEDATA/deepNF", type=str)
parser.add_argument('-a', '--architecture', default="2", type=int)
parser.add_argument('-n', '--n-trials', default=10, type=int)
parser.add_argument('-v', '--validation', default='cv', type=str)
parser.add_argument('--tags', default="", type=str)
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

# # Validation type
validation_types = {
    'cv': ('P_3', 'P_2', 'P_1', 'F_3', 'F_2', 'F_1', 'C_3', 'C_2', 'C_1'),
    'cv2': ('P_1', 'P_2', 'P_3', 'F_1', 'F_2', 'F_3', 'C_1', 'C_2', 'C_3')}

try:
    annotation_types = validation_types[validation]
except KeyError as err:
    err.args = (f'Not a valid validation type: {validation}',)


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

    results_summary_file = os.path.join(
        results_path,
        f'{model_name}_{validation}_performance.txt')

    with open(results_summary_file, 'w') as fout:
        for level in annotation_types:
            stdout('Running for level', level)
            if validation == 'cv':
                perf = cross_validation(
                    embeddings,
                    GO[level],
                    n_trials=n_trials,
                    fname=os.path.join(
                        results_path,
                        f'{model_name}_{level}_{validation}_performance_trials.txt'))

                fout.write(f'\n{level}\n')

                for m in measures:
                    fout.write(f'{m} {perf[m]:.5f}\n')
                fout.write('\n')


if __name__ == '__main__':
    main()
