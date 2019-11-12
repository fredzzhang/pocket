"""
Ranked score plot with extended information

Written by Frederic Zhang
Australian National University

Last updated in Jul. 2019
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from .ranked_score_plot import RankedScorePlot
from ..utils import InferenceManager, load_pkl_from

class ExtendedRSP(RankedScorePlot):
    """
    Extended ranked score plots

    Allows plotting up to 3 additional attributes of data, besides the ranked scores.

    Arguments:
        num_classes(int): Number of classes
        markers(list[str]): Markers for different attributes
        kwargs(dict): Keyworded attributes e.g. {name0: attr0, ...}
            name0(str): Attribute name
            attr0(ndarray[N]): Attribute data. N should be equal to the total
                number of samples (including all positives and negatives)
    """
    def __init__(self, num_classes, markers=['+'], **kwargs):
        super(ExtendedRSP, self).__init__(num_classes)
        assert len(kwargs) <= len(markers),\
                'Not enough marker types provided'
        self._markers = markers
        self._attr = kwargs

    def show(self, cls, order='default'):
        """
        Show the ranked score plot with additional attributes

        Arguments:
            cls(int): Class index
            order(str): The order by which samples are ranked
                'default': positives in descending order, negatives in ascending order
                'reverse': positives in ascending order, negatives in descending order
        """
        assert type(cls) is int,\
                'Class index has to be an integer'
        assert cls >= 0 and cls < self._num_classes,\
                'Class index has to be between 0 and {NUM_CLASSES}'

        if order == 'default':
            x1 = np.linspace(1, 0, len(self._pool[cls][0]))
            x2 = np.linspace(0, 1, len(self._pool[cls][1]))
            x3 = np.linspace(0, 1, len(self._pool[cls][2]))
        elif order == 'reverse':
            x1 = np.linspace(1e-5, 1, len(self._pool[cls][0]))
            x2 = np.linspace(1, 1e-5, len(self._pool[cls][1]))
            x3 = np.linspace(1, 1e-5, len(self._pool[cls][2]))
            plt.xscale('log')
        else:
            raise ValueError('Unsupported ranking method \"{}\"'.format(order))

        # plot additional attributes
        for i, (k, v) in enumerate(self._attr.items()):
            plt.plot(x3, v[self._pool[cls][2][:, 0].astype(np.int)], 'r', alpha=0.25,
                    linestyle='None', markersize=5, marker=self._markers[i], label=k)
            plt.plot(x2, v[self._pool[cls][1][:, 0].astype(np.int)], 'b', alpha=0.35,
                    linestyle='None', markersize=4, marker=self._markers[i], label=k)
            plt.plot(x1, v[self._pool[cls][0][:, 0].astype(np.int)], 'g', alpha=0.8,
                    linestyle='None', markersize=4, marker=self._markers[i], label=k)

        # plot positives in green
        plt.plot(x1, self._pool[cls][0][:, 1], 'g', marker='d', markersize=6, label='Positives')
        # plot type-I negatives in blue
        plt.plot(x2, self._pool[cls][1][:, 1], 'b', label='Type-I Negatives')
        # plot type-II negatives in red
        plt.plot(x3, self._pool[cls][2][:, 1], 'r', label='Type-II Negatives')

        plt.legend()
        plt.grid()
        plt.title('Ranked Score Plot for Class {}'.format(cls))
        plt.xlabel('Normalized sample indices')
        plt.ylabel('Scores')
        plt.show()

    def save(self, cache_dir, order='default'):
        """
        Save the ranked score plots

        Arguments:
             cache_dir(str): cache directory
             order(str): The order by which samples are ranked
                 'default': positives in descending order, negatives in ascending order
                 'reverse': positives in ascending order, negatives in descending order
        """
        for cls in range(self._num_classes):

            if order == 'default':
                x1 = np.linspace(1, 0, len(self._pool[cls][0]))
                x2 = np.linspace(0, 1, len(self._pool[cls][1]))
                x3 = np.linspace(0, 1, len(self._pool[cls][2]))
            elif order == 'reverse':
                # for log plot, x index starts from a small value above zero
                x1 = np.linspace(1e-5, 1, len(self._pool[cls][0]))
                x2 = np.linspace(1, 1e-5, len(self._pool[cls][1]))
                x3 = np.linspace(1, 1e-5, len(self._pool[cls][2]))
                plt.xscale('log')
            else:
                raise ValueError('Unsupported ranking methods {}'.format(order))

            for i, (k, v) in enumerate(self._attr.items()):
                plt.plot(x3, v[self._pool[cls][2][:, 0].astype(np.int)], 'r', alpha=0.25,
                        linestyle='None', markersize=5, marker=self._markers[i], label=k)
                plt.plot(x2, v[self._pool[cls][1][:, 0].astype(np.int)], 'b', alpha=0.35,
                        linestyle='None', markersize=4, marker=self._markers[i], label=k)
                plt.plot(x1, v[self._pool[cls][0][:, 0].astype(np.int)], 'g', alpha=0.8,
                        linestyle='None', markersize=4, marker=self._markers[i], label=k)

            plt.plot(x1, self._pool[cls][0][:, 1], 'g', marker='d', markersize=6, label='Positives')
            plt.plot(x2, self._pool[cls][1][:, 1], 'b', label='Type-I Negatives')
            plt.plot(x3, self._pool[cls][2][:, 1], 'r', label='Type-II Negatives')

            plt.legend()
            plt.grid()
            plt.title('Ranked Score Plot for Class {}'.format(cls))
            plt.xlabel('Normalized sample indices')
            plt.ylabel('Classification scores')
            plt.savefig(os.path.join(cache_dir, 'Class_{}.png'.format(cls)))
            plt.clf()

def compute_extended_rsp(
        net,
        labels,
        dataloader,
        device='cpu',
        cache_dir='./',
        multi_gpu=False,
        input_transform=lambda a: a,
        output_transform=lambda a: torch.cat([b for b in a], 0),
        **kwargs):
        """
        Compute the ranked scores for three types of samples across all classes
            Positive
            Type-I Negative
            Type-II Negative

        Arguments:

        [REQUIRED ARGS]
            net(Module): Network model
            labels(ndarray[N, c] or Tensor[N, c]): Labels for all the data in loader, where N is the 
                number of samples and C is the number of classes. NOTE: The order of the labels
                should be consistent with order of samples in dataloader
            dataloader(iterable): Dataloader with batch data as [INPUT_1, ..., INPUT_N],
                where each input should be a Tensor. Otherwise, use {input_transform}
                to format the batch
    
        [OPTIONAL ARGS]
            device(str): Primary device, has to be cuda:0 when using multiple gpus
            multi_gpu(bool): When True, use all visible gpus in forward pass
            cache_dir(str): Directory to save cache
            input_transform(callable): Transform the batch data to desired format
            output_transform(callable): Transform the collective output to a single Tensor with
                size NxC as the labels
            kwargs(dict): Keyworded attributes. Refer to class ExtendedRSP
        """
        if os.path.exists(os.path.join(cache_dir, 'output.pkl')):
            output = load_pkl_from(os.path.join(cache_dir, 'output.pkl'))
        else:
            inference = InferenceManager(
                    net,
                    dataloader,
                    device,
                    cache_dir,
                    multi_gpu,
                    True,
                    100,
                    input_transform)
            output = inference()
        output = output_transform(output)

        ranked_score = ExtendedRSP(labels.shape[1], **kwargs)
        ranked_score.push(output, labels)

        return ranked_score
