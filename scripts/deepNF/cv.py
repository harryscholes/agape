'''
Generate embeddings from the middle layer of the autoencoder and use these to
train a classifier with k-fold cross-validation.
'''
import os
import argparse
import glob
from keras.models import load_model
from sklearn.preprocessing import minmax_scale
import scipy.io as sio
from agape.deepNF.validation import cross_validation
from agape.deepNF.utils import mkdir, plot_loss, load_ppmi_matrices
from agape.utils import stdout

print(__doc__)

##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--models-path', default="./models", type=str)
parser.add_argument('-r', '--results-path', default="./results", type=str)
parser.add_argument('-d', '--data-path', default="$AGAPEDATA/deepNF", type=str)
parser.add_argument('-a', '--architecture', default="2", type=int)
# parser.add_argument('-b', '--batch-size', default=128, type=int)
# parser.add_argument('-n', '--n-trials', default=10, type=int)
parser.add_argument('-v', '--validation', default='cv', type=str)
parser.add_argument('--tags', default="", type=str)
args = parser.parse_args()

stdout("Command line arguments", args)

models_path = os.path.expandvars(args.models_path)
results_path = os.path.expandvars(args.results_path)
data_path = os.path.expandvars(args.data_path)
architecture = args.architecture
# batch_size = args.batch_size
# n_trials = args.n_trials
# tags = args.tags
validation = args.validation

# # Validation type
# validation_types = {
#     'cv': ('P_3', 'P_2', 'P_1', 'F_3', 'F_2', 'F_1', 'C_3', 'C_2', 'C_1'),
#     'cv2': ['P_1', 'P_2', 'P_3', 'F_1', 'F_2', 'F_3', 'C_1', 'C_2', 'C_3']}
#
# try:
#     annotation_types = validation_types[validation]
# except KeyError as err:
#     err.args = (f'Not a valid validation type: {validation}',)
#
# # Performance measures
# measures = ['m-aupr_avg', 'm-aupr_std', 'M-aupr_avg', 'M-aupr_std',
#             'F1_avg', 'F1_std', 'acc_avg', 'acc_std']


#################
# Load networks #
#################

networks, dims = load_ppmi_matrices(data_path)


###############
# Load models #
###############

model_names = sorted(glob.glob(os.path.join(os.path.expandvars(models_path),
                                            'yeast_MDA_arch_*.h5')))

stdout("Model names", model_names)


# For now I am only loading a single model at a time, specified by
# `--architecture`. TODO improve this to handle multiple models at one time.
if architecture:
    for m in model_names:
        if int(m[-4]) == architecture:
            mid_model = load_model(m)
            model_name = os.path.basename(m).split(".")[0]
else:
    raise Warning("`--architecture` must be supplied")


###############################
# Generate network embeddings #
###############################

results_summary_file = os.path.join(
    results_path,
    f'{model_name}_{validation}_performance.txt')


with open(results_summary_file, 'w') as fout:
    stdout("Running for", model_name)
    fout.write(f'\n{model_name}\n')

    embeddings = minmax_scale(mid_model.predict(networks))

    embeddings_path = os.path.join(
        os.path.expandvars(results_path),
        f'{model_name}_features.mat')

    stdout("Saving embeddings to", embeddings_path)

    sio.savemat(embeddings_path, {'embeddings': embeddings})

    # for level in annotation_types:
    #     print(f"Running for level: {level}")
    #     if validation == 'cv':
    #         perf = cross_validation(
    #             embeddings,
    #             GO[level],
    #             n_trials=n_trials,
    #             fname=os.path.join(
    #                 results_path,
    #                 f'{model_name}_{level}_{validation}_performance_trials.txt'))
    #
    #         fout.write(f'\n{level}\n')
    #
    #         for m in measures:
    #             fout.write(f'{m} {perf[m]:.5f}\n')
    #         fout.write('\n')
