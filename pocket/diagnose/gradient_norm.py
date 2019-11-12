"""
Compute and visualize gradient norm

Written by Frederic Zhang
Australian National University

Last updated in Aug. 2019
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from .ranked_score_plot import RankedScorePlot
from ..utils import InferenceManager, load_pkl_from

class GradientNorm(RankedScorePlot):
    r"""
    Class for gradient norm
    """
    def __init__(self, num_classes):
        super(GradientNorm, self).__init__(num_classes)

    def show(self, cls=-1):
        """
        Show the gradient norm

        Arguments:
            cls(int): Class index. Use -1 to show combine all classes   
        """
        assert type(cls) is int,\
                'Class index has to be an integer'
        assert (cls >= 0 and cls < self._num_classes) or cls == -1,\
                'Class index has to be -1 or between 0 and {NUM_CLASSES}'

        plt.figure(figsize=(20, 4))
        if cls == -1:
            raise NotImplementedError
        else:
            gn = np.hstack([
                1 - self._pool[cls][0][:, 1],
                self._pool[cls][1][:, 1],
                self._pool[cls][2][:, 1]
                ])
            inds = np.argsort(gn)
            order = np.argsort(inds)
            x = np.linspace(1e-4, 1, len(gn))

            mark1 = len(self._pool[cls][0])
            mark2 = len(self._pool[cls][0]) + len(self._pool[cls][1])

            # plot type-II negatives in red
            i3 = order[mark2:]
            plt.plot(x[i3], gn[mark2:], 'rx',
                    linestyle='None', markersize=5, alpha=0.25, label='Type-II Negatives')

            # plot type-I negatives in blue
            i2 = order[mark1: mark2]
            plt.plot(x[i2], gn[mark1: mark2], 'b|',
                    linestyle='None', markersize=5, alpha=0.2, label='Type-I Negatives')

            # plot positives in green
            i1 = order[:mark1]
            plt.plot(x[i1], gn[:mark1], 'gx',
                    linestyle='None', markersize=5, alpha=0.8, label='Positives')

        plt.legend()
        plt.grid()
        plt.title('Gradient Norm for class {}'.format(cls))
        plt.xlabel('Normalized sample indices')
        plt.ylabel('Gradient Norm')
        plt.tight_layout()
        plt.show()

    def save(self, cache_dir):
        raise NotImplementedError
