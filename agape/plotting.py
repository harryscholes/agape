'''Plotting functions.
'''
import os
import numpy as np
from typing import Dict
import matplotlib as mpl

try:
    assert 'DISPLAY' in os.environ
except AssertionError:
    mpl.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

__all__ = ['plot_loss']


def plot_loss(plot_data: Dict[str, dict], filename: str,
              end_epoch: int = None, log_y: bool = False,
              colors=sns.color_palette('colorblind')):
    '''Plot the training and validation loss from Keras training histories.

    # Arguments
        plot_data: Dict[str, dict], dict of Keras training history dicts keyed
            by a str to label the series in the figure legend
        filename: str, file name
        end_epoch: int, final epoch to plot
    '''
    def plotter(history, label, c, n=None):
        '''Plots the lines.
        '''
        ax.plot(np.arange(1, len(history['val_loss']) + 1)[:n],
                np.array(history['val_loss'])[:n],
                color=c, label=label)
        ax.plot(np.arange(1, len(history['loss']) + 1)[:n],
                np.array(history['loss'])[:n],
                color=c, linestyle='--', alpha=0.5)

    fig, ax = plt.subplots(1, figsize=(4, 3))

    for idx, (label, history) in enumerate(plot_data.items()):
        plotter(history, label, colors[idx], end_epoch)

    if log_y:
        ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(f'{filename}.pdf')
