import os
import glob
from scipy import io
from pathlib import Path
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from agape.utils import stdout
from sklearn.preprocessing import minmax_scale
import pandas as pd
import numpy as np
import seaborn as sns


def mkdir(directory):
    '''Make a directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_loss(history, models_path, model_name):
    '''Plot autoencoder training loss.
    '''
    sns.set(context='paper', style='ticks')
    plt.plot(history.history['loss'], 'o-')
    plt.plot(history.history['val_loss'], 'o-')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(str(Path(models_path, model_name + '_loss.png')),
                bbox_inches='tight')


def _load_ppmi_matrix(filepath):
    '''Load a PPMI matrix of a network adjacency matrix.

    # Arguments:
        filepath: str, path to .mat file

    # Returns
        M: numpy.ndarray, PPMI matrix

    # Raises
        OSError: if .mat file does not exist at `filepath`
    '''
    if not os.path.exists(filepath):
        raise OSError("Network not found at:", filepath)

    print(f"Loading network from {filepath}")
    M = io.loadmat(filepath, squeeze_me=True)['Net'].toarray()
    return M


def load_ppmi_matrices(data_path):
    '''Load PPMI matrices.

    # Arguments
        data_path: str, path to .mat files

    # Returns
        Ms: List[numpy.ndarray], PPMI matrices
        dims: List[int], dimensions of matrices
    '''
    paths = sorted(glob.glob(os.path.join(data_path, "*.mat")))
    stdout('Networks', paths)

    Ms = []
    for p in paths:
        M = _load_ppmi_matrix(p)
        Ms.append(minmax_scale(M))

    dims = [i.shape[1] for i in Ms]
    stdout('Input dims', dims)
    return Ms, dims


def gene2index(mapping_file=None):
    '''Returns a dictionary mapping genes to PPMI matrix indicies.

    # Arguments
        mapping_file: str, path to mapping file

    # Returns
        d: dict, mapper

    # Raises
        FileNotFoundError: if `mapping_file` not found
    '''
    if mapping_file is None:
        mapping_file = os.path.join(
            os.path.expandvars('$AGAPEDATA'),
            'deepNF', 'networks', 'yeast_net_genes.csv')
    try:
        df = pd.read_csv(mapping_file, header=None, index_col=0)
    except FileNotFoundError:
        raise

    d = df[1].to_dict()
    return d


def load_embeddings(embeddings_file: str) -> np.ndarray:
    '''Load embeddings from file.

    # Arguments
        embeddings_file: str, path to embeddings file *_features.mat file

    # Returns
        embeddings: np.ndarray, node embeddings

    # Raises
        FileNotFoundError: if `embeddings_file` does not exist
    '''
    try:
        M = io.loadmat(embeddings_file, squeeze_me=True)['embeddings']
        return M
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Embeddings not found at {embeddings_file}') from None
