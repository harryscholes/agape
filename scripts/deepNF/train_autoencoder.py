'''
Train a multimodal deep autoencoder on multiple networks.

Usage:
    python train_autoencoder.py
'''
import os
# os.environ["KERAS_BACKEND"] = "tensorflow"
import pickle
import argparse
from agape.deepNF.utils import mkdir, load_ppmi_matrices
from agape.utils import stdout
from agape.ml.autoencoder import MultimodalAutoencoder
from sklearn.preprocessing import minmax_scale
from scipy import io as sio
from agape.plotting import plot_loss

##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--organism', default='yeast', type=str)
parser.add_argument('-t', '--model-type', default='mda', type=str)
parser.add_argument('-m', '--models-path', default="models", type=str)
parser.add_argument('-d', '--data-path', default="$AGAPEDATA/deepNF", type=str)
parser.add_argument('-l', '--layers', type=str)
parser.add_argument('-a', '--activation', default="selu", type=str)
parser.add_argument('-z', '--optimizer', default="AdaMax", type=str)
parser.add_argument('-e', '--epochs', default=10, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--outfile-tags', default="", type=str)
args = parser.parse_args()

stdout("Command line arguments", args)

org = args.organism
model_type = args.model_type
models_path = os.path.expandvars(args.models_path)
data_path = os.path.expandvars(args.data_path)
layers = [int(i) for i in args.layers.split('-')]
activation = args.activation
optimizer = args.optimizer
epochs = args.epochs
batch_size = args.batch_size
ofile_tags = args.outfile_tags


########
# defs #
########

def main():
    ######################
    # Prepare filesystem #
    ######################

    mkdir("models")

    #################
    # Load networks #
    #################

    networks, dims = load_ppmi_matrices(data_path)

    #########################
    # Train the autoencoder #
    #########################

    model_name = [f'{org}', f'{model_type.upper()}', f'arch_{args.layers}']

    if ofile_tags != '':
        model_name.append(f'{f"{ofile_tags}" if ofile_tags != "" else ""}')

    model_name = '_'.join(model_name)

    stdout("Running for architecture", model_name)

    autoencoder = MultimodalAutoencoder(
        x_train=networks,
        x_val=0.1,
        layers=layers,
        epochs=epochs,
        batch_size=batch_size,
        activation=activation,
        optimizer=optimizer,
        early_stopping=(5, 0.),
        verbose=2)

    autoencoder.train()

    history = autoencoder.history.history

    with open(os.path.join(models_path, f'{model_name}_training_history.pkl'),
              'wb') as f:
        pickle.dump(history, f)

    plot_loss({model_name: history}, f'{models_path}/{model_name}')

    embeddings = minmax_scale(autoencoder.encode(networks))

    embeddings_path = os.path.join(
        models_path, f'{model_name}_embeddings.mat')

    sio.savemat(embeddings_path, {'embeddings': embeddings})


###################
# Run the program #
###################

if __name__ == '__main__':
    print(__doc__)
    main()
