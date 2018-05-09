import os
from pathlib import Path
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def mkdir(directory):
    '''Make a directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_loss(history, models_path, model_name):
    '''Plot autoencoder training loss.
    '''
    plt.plot(history.history['loss'], 'o-')
    plt.plot(history.history['val_loss'], 'o-')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(str(Path(models_path, model_name + '_loss.png')),
                bbox_inches='tight')
