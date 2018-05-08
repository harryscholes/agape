'''
Run deepNF

Code originally by Vladimir Gligorijevi, adapted from
https://github.com/VGligorijevic/deepNF.

To run:
    python deepNF.py

    Legion:

    qsub -b y -N deepNF_GPU -pe smp 8 -l h_rt=2:0:0,mem=15G,gpu=1 -ac allow=P \
        python $HOME/git/deepNF/deepNF.py --architecture 2

    qsub -b y -N deepNF -pe smp 8 -l h_rt=2:0:0,mem=15G \
        python $HOME/git/deepNF/deepNF.py --architecture 2

    CS:

    qsub -b y -N deepNF_GPU
        -pe smp 8 -l h_rt=2:0:0,h_vmem=7.8G,tmem=7.8G,gpu=1 \
        -ac allow=P \
        python $HOME/git/deepNF/deepNF.py --architecture 2

    qsub -b y -N deepNF -pe smp 2 -l h_rt=2:0:0,h_vmem=7.8G,tmem=7.8G \
        python $HOME/git/deepNF/deepNF.py --architecture 2


    python deepNF.py -d $AGAPEDATA/deepNF/test
'''
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from pathlib import Path
from keras.models import Model, load_model
from sklearn.preprocessing import minmax_scale
from keras.callbacks import EarlyStopping
import pickle
import sys
import argparse
import scipy.io as sio
import glob

from validation import cross_validation, temporal_holdout
from autoencoders import MDA, AE
from utils import mkdir, plot_loss


##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--organism',     default='yeast',      type=str)
parser.add_argument('-m', '--model-type',   default='mda',        type=str)
parser.add_argument('-p', '--models-path',  default="./models",   type=str)
parser.add_argument('-r', '--results-path', default="./results",  type=str)
parser.add_argument('-d', '--data-path',    default="$AGAPEDATA/deepNF", type=str)
parser.add_argument('-a', '--architecture', default="2",          type=str)
parser.add_argument('-e', '--epochs',       default=10,           type=int)
parser.add_argument('-b', '--batch-size',   default=128,          type=int)
parser.add_argument('-n', '--n-trials',     default=10,           type=int)
parser.add_argument(      '--K',            default=3,            type=int)
parser.add_argument(      '--alpha',        default=.98,          type=float)
parser.add_argument(      '--outfile-tags', default="",           type=str)
parser.add_argument('-v', '--validation', default='cv', type=str)
args = parser.parse_args()

print("ARGUMENTS\n", args)

org          = args.organism
model_type   = args.model_type
models_path  = args.models_path
results_path = args.results_path
data_path    = os.path.expandvars(args.data_path)
select_arch  = [int(i) for i in args.architecture.split(",")]
epochs       = args.epochs
batch_size   = args.batch_size
n_trials     = args.n_trials
K            = args.K
alpha        = args.alpha
ofile_tags   = args.outfile_tags
validation   = args.validation

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

# Validation type
validation_types = {
    'cv': ['level1', 'level2', 'level3'],
    'th': ['MF', 'BP', 'CC']}

try:
    annotation_types = validation_types[validation]
except KeyError as err:
    err.args = (f'Not a valid validation type: {validation}',)


########
# defs #
########

def main():
    ######################
    # Prepare filesystem #
    ######################

    mkdir("models")
    mkdir("results")

    #################
    # Load networks #
    #################

    def load_network(filepath):
        if not os.path.exists(filepath):
            raise OSError("Network not found at:", filepath)

        print(f"Loading network from {filepath}")
        N = sio.loadmat(filepath, squeeze_me=True)['Net'].toarray()
        return N

    network_paths = glob.glob(os.path.join(data_path, "*.mat"))
    print(network_paths)

    basename = os.path.basename(network_paths[0]).split("_")[0]

    networks = []

    for network_path in network_paths:
        network = load_network(network_path)
        networks.append(minmax_scale(network))

    input_dims = [i.shape[1] for i in networks]

    print(input_dims)

    #########################
    # Train the autoencoder #
    #########################

    model_names = []

    for arch in architectures:
        model_name = f"{basename}_{model_type.upper()}_arch_{str(arch)}{f'_{ofile_tags}' if ofile_tags != '' else ''}"
        model_names.append(model_name)
        print(f"Running for architecture: {model_name}")

        # Build model
        model = MDA(input_dims, architectures[arch])

        # Train model
        history = model.fit(
            networks,
            networks,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.0001,
                    patience=2)])

        plot_loss(history, models_path, model_name)

        with open(Path(models_path, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)

        # Extract middle layer
        mid_model = Model(
            inputs=model.input,
            outputs=model.get_layer('middle_layer').output)

        mid_model.save(Path(models_path, f"{model_name}.h5"))

    ###############################
    # Generate network embeddings #
    ###############################

    for model_name in model_names:
        print(f"Running for: {model_name}")

        my_file = Path(models_path, f"{model_name}.h5")

        if my_file.exists():
            mid_model = load_model(my_file)
        else:
            raise OSError("Model does not exist", my_file)

        embeddings = minmax_scale(mid_model.predict(networks))

        sio.savemat(
            str(Path(results_path, model_name + '_features.mat')),
            {'embeddings': embeddings})


###################
# Run the program #
###################

if __name__ == '__main__':
    main()
