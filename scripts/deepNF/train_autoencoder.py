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

print(__doc__)

##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--organism', default='yeast', type=str)
parser.add_argument('-t', '--model-type', default='mda', type=str)
parser.add_argument('-m', '--models-path', default="./models", type=str)
parser.add_argument('-d', '--data-path', default="$AGAPEDATA/deepNF", type=str)
parser.add_argument('-a', '--architecture', default="2", type=str)
parser.add_argument('-e', '--epochs', default=10, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--outfile-tags', default="", type=str)
args = parser.parse_args()

stdout("Command line arguments", args)

org = args.organism
model_type = args.model_type
models_path = os.path.expandvars(args.models_path)
data_path = os.path.expandvars(args.data_path)
select_arch = [int(i) for i in args.architecture.split(",")]
epochs = args.epochs
batch_size = args.batch_size
ofile_tags = args.outfile_tags

# Autoencoder architecture
architectures_dict = {
    1: [600],
    2: [6*2000, 600, 6*2000],
    3: [6*2000, 6*1000, 600, 6*1000, 6*2000],
    4: [6*2000, 6*1000, 6*500, 600, 6*500, 6*1000, 6*2000],
    5: [6*2000, 6*1200, 6*800, 600, 6*800, 6*1200, 6*2000],
    6: [6*2000, 6*1200, 6*800, 6*400, 600, 6*400, 6*800, 6*1200, 6*2000]}

architectures = {k: v for k, v in architectures_dict.items()
                 if k in select_arch}


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

    model_names = []

    for arch in architectures:
        model_name = f"{org}_{model_type.upper()}_arch_{str(arch)}{f'_{ofile_tags}' if ofile_tags != '' else ''}"
        model_names.append(model_name)
        stdout("Running for architecture", model_name)

        # Build model
        model = MDA(dims, architectures[arch])

        # Train model
        history = model.fit(
            networks,
            networks,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1,
            verbose=2,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.0001,
                    patience=2)])

        plot_loss(history, models_path, model_name)

        with open(Path(models_path,
                       f'{model_name}_training_history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)

        # Extract middle layer
        mid_model = Model(
            inputs=model.input,
            outputs=model.get_layer('middle_layer').output)

        mid_model.save(Path(models_path, f"{model_name}.h5"))


###################
# Run the program #
###################

if __name__ == '__main__':
    main()
