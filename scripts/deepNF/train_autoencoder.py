'''
Train a multimodal deep autoencoder on multiple networks.

Usage:
    python train_autoencoder.py
'''
import os
# os.environ["KERAS_BACKEND"] = "tensorflow"
from pathlib import Path
from keras.models import Model
from keras.callbacks import EarlyStopping
import pickle
import argparse
from agape.deepNF.autoencoders import MDA
from agape.deepNF.utils import mkdir, plot_loss, load_ppmi_matrices
from agape.utils import stdout
from agape.ml.autoencoder import MultimodalAutoencoder
from keras.optimizers import SGD
from sklearn.preprocessing import minmax_scale

print(__doc__)

##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--organism', default='yeast', type=str)
parser.add_argument('-t', '--model-type', default='mda', type=str)
parser.add_argument('-m', '--models-path', default="models", type=str)
parser.add_argument('-d', '--data-path', default="$AGAPEDATA/deepNF", type=str)
parser.add_argument('-l', '--layers', type=str)
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

    model_name = f"{org}_{model_type.upper()}_arch_{args.layers}{f'_{ofile_tags}' if ofile_tags != '' else ''}"

    stdout("Running for architecture", model_name)

    autoencoder = MultimodalAutoencoder(
        x_train=networks,
        x_val=0.1,
        layers=layers,
        epochs=epochs,
        batch_size=batch_size,
        activation='sigmoid',
        optimizer=SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False),
        verbose=2)

    autoencoder.train()

    with open(os.path.join(models_path, f'{model_name}_training_history.pkl'),
              'wb') as f:
        pickle.dump(autoencoder.history.history, f)

    plot_loss(autoencoder.history, models_path, model_name)

    autoencoder.encoder.save(os.path.join(models_path, f"{model_name}.h5"))

    embeddings = minmax_scale(autoencoder.predict(networks))

    embeddings_path = os.path.join(
        models_path, f'{model_name}_embeddings.mat')

    sio.savemat(embeddings_path, {'embeddings': embeddings})


###################
# Run the program #
###################

if __name__ == '__main__':
    main()
