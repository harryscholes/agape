'''
Generate embeddings from the middle layer of the autoencoder.
'''
import os
import argparse
import glob
from keras.models import load_model
from sklearn.preprocessing import minmax_scale
import scipy.io as sio
from agape.deepNF.utils import load_ppmi_matrices
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
parser.add_argument('-v', '--validation', default='cv', type=str)
parser.add_argument('--tags', default="", type=str)
args = parser.parse_args()

stdout("Command line arguments", args)

models_path = os.path.expandvars(args.models_path)
results_path = os.path.expandvars(args.results_path)
data_path = os.path.expandvars(args.data_path)
architecture = args.architecture
validation = args.validation


def main():
    #################
    # Load networks #
    #################

    networks, dims = load_ppmi_matrices(data_path)

    ###############
    # Load models #
    ###############

    model_names = sorted(glob.glob(
        os.path.join(os.path.expandvars(models_path),
                     'yeast_MDA_arch_*.h5')))

    stdout("Model names", model_names)

    # For now I am only loading a single model at a time, specified by
    # `--architecture`. TODO improve this to handle multiple models at one
    # time.
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
    stdout("Running for", model_name)

    embeddings = minmax_scale(mid_model.predict(networks))

    embeddings_path = os.path.join(
        os.path.expandvars(results_path),
        f'{model_name}_features.mat')

    stdout("Saving embeddings to", embeddings_path)

    sio.savemat(embeddings_path, {'embeddings': embeddings})


if __name__ == '__main__':
    main()
