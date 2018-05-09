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
# os.environ["KERAS_BACKEND"] = "tensorflow"
from pathlib import Path
from keras.models import Model, load_model
from sklearn.preprocessing import minmax_scale
from keras.callbacks import EarlyStopping
import pickle
import argparse
import scipy.io as sio
import glob
from validation import cross_validation
from autoencoders import MDA
from utils import mkdir, plot_loss


##########################
# Command line arguments #
##########################

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--organism', default='yeast', type=str)
parser.add_argument('-t', '--model-type', default='mda', type=str)
parser.add_argument('-m', '--models-path', default="./models", type=str)
parser.add_argument('-r', '--results-path', default="./results", type=str)
parser.add_argument('-d', '--data-path', default="$AGAPEDATA/deepNF", type=str)
parser.add_argument('-p', '--precalculated-embeddings', default=False, type=bool)
parser.add_argument('-a', '--architecture', default="2", type=str)
parser.add_argument('-e', '--epochs', default=10, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-n', '--n-trials', default=10, type=int)
parser.add_argument('-v', '--validation', default='cv', type=str)
parser.add_argument('--K', default=3, type=int)
parser.add_argument('--alpha', default=.98, type=float)
parser.add_argument('--outfile-tags', default="", type=str)
args = parser.parse_args()

print("ARGUMENTS\n", args)

org = args.organism
model_type = args.model_type
models_path = os.path.expandvars(args.models_path)
results_path = os.path.expandvars(args.results_path)
data_path = os.path.expandvars(args.data_path)
precalculated_embeddings = args.precalculated_embeddings
select_arch = [int(i) for i in args.architecture.split(",")]
epochs = args.epochs
batch_size = args.batch_size
n_trials = args.n_trials
K = args.K
alpha = args.alpha
ofile_tags = args.outfile_tags
validation = args.validation

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
    'cv': ('P_3', 'P_2', 'P_1', 'F_3', 'F_2', 'F_1', 'C_3', 'C_2', 'C_1'),
    'cv2': ['P_1', 'P_2', 'P_3', 'F_1', 'F_2', 'F_3', 'C_1', 'C_2', 'C_3']}

try:
    annotation_types = validation_types[validation]
except KeyError as err:
    err.args = (f'Not a valid validation type: {validation}',)

# Performance measures
measures = ['m-aupr_avg', 'm-aupr_std', 'M-aupr_avg', 'M-aupr_std',
            'F1_avg', 'F1_std', 'acc_avg', 'acc_std']


########
# defs #
########

def main():
    ######################
    # Prepare filesystem #
    ######################

    mkdir("models")
    mkdir("results")

    #######################
    # Load GO annotations #
    #######################

    if validation == 'cv':
        annotation_dir = os.path.join(
            os.path.expandvars('$AGAPEDATA'), 'annotations')
        annotation_file = 'yeast_annotations.mat'
        print('Loading GO annotations')

        GO = sio.loadmat(
            os.path.join(annotation_dir, annotation_file))

    #################
    # Load networks #
    #################

    def load_network(filepath):
        '''Load a network at `filepath`.
        '''
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

    if not precalculated_embeddings:
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

    elif precalculated_embeddings:
        model_names = ["yeast_MDA_arch_2"]  # TODO remove hardcoded path
        mid_model = load_model(os.path.expandvars(f"$AGAPEDATA/{model_names[0]}.h5"))

    ###############################
    # Generate network embeddings #
    ###############################

    results_summary_file = os.path.join(
        results_path,
        f'deepNF_{model_type}_{"{ofile_tags}_" if ofile_tags != "" else ""}{validation}_performance_{org}.txt')

    with open(results_summary_file, 'w') as fout:

        for model_name in model_names:
            print(f"Running for: {model_name}")
            fout.write(f'\n{model_name}\n')

            my_file = Path(models_path, f"{model_name}.h5")

            if my_file.exists():
                mid_model = load_model(my_file)
            else:
                raise OSError("Model does not exist", my_file)

            embeddings = minmax_scale(mid_model.predict(networks))

            sio.savemat(
                str(Path(results_path, model_name + '_features.mat')),
                {'embeddings': embeddings})

            for level in annotation_types:
                print(f"Running for level: {level}")
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


###################
# Run the program #
###################

if __name__ == '__main__':
    main()
