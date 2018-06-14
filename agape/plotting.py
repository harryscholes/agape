'''Plotting functions.
'''
import os
import numpy as np
import seaborn as sns
from typing import Tuple, List

try:
    assert 'DISPLAY' in os.environ
except AssertionError:
    import matplotlib
    matplotlib.use('pdf')
finally:
    import matplotlib.pyplot as plt

__all__ = ['plot_loss']


def plot_loss(plot_data: List[Tuple[dict, str]], filename: str,
              end_epoch: int = None):
    '''Plot the training and validation loss from Keras training histories.

    # Arguments
        plot_data: List[Tuple[dict, str]], list of tuples of the form
            (history, label) i.e. Keras training history dict and str to label
            the series in the figure legend
        filename: str, file name
        end_epoch: int, final epoch to plot
    '''
    plt.ioff()

    def plotter(history, label, c, n=None):
        '''Plots the lines.
        '''
        ax.plot(np.arange(1, len(history['val_loss']) + 1)[:n],
                np.array(history['val_loss'])[:n],
                color=c, label=label)
        ax.plot(np.arange(1, len(history['loss']) + 1)[:n],
                np.array(history['loss'])[:n],
                color=c, linestyle='--', alpha=0.5)

    c = sns.color_palette('colorblind')
    fig, ax = plt.subplots(1, figsize=(4, 3))

    for idx, (history, label) in enumerate(plot_data):
        plotter(history, label, c[idx], end_epoch)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(f'{filename}.pdf')
