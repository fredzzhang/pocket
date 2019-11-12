"""
Utilities for ranked score plots

Written by Frederic Zhang
Australian National University

Last updated in Jun. 2019
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from ..utils import InferenceManager, load_pkl_from

class RankedScorePlot:
    r"""
    Ranked score plots

    For each class, three types of samples will be plotted
        positives: positive samples for current class
        type-I negatives: positive samples for other classes but negative for current class
        type-II negatives: negative samples for all classes

    The vertical axis indicates the classification score for a sample, while the horizontal
    axis shows the normalized index (between 0 and 1) of a sample. Samples will be ranked based
    on their scores. By default, positives are in descending order while type-I and type-II 
    negatives will be in ascending order

    Arguments:
        num_classes(int): number of classes

    Example:

        >>> import torch
        >>> from pocket.diagnose import RankedScorePlot
        >>> # Predictions contain 3 samples on 2 classes
        >>> pred = torch.tensor([[0.1, 0.9], [0.2, 0.7], [0.4, 0.9]])
        >>> labels = torch.tensor([[0., 1.], [1., 0.], [1., 0.]])
        >>> # Initialize ranked score plot with 2 classes
        >>> rsp = RankedScorePlot(2)
        >>> # Update ranked score plot
        >>> rsp.push(pred, labels)
        >>> # Print data for class 0
        >>> # In order, there are 2 positive samples (sample #1 and sample #2)
        >>> # 1 type-I negative (sample #0) and 0 type-II negative
        >>> rsp.fetch_class_data(0)
        [array([[1., 0.2],
                [2., 0.4]]),
         array([[0., 0.1]]),
         array([], shape=(0, 2), dtype=float64)]
        >>> # Print data for class 1
        >>> rsp.fetch_class_data(1)
        [array([[0., 0.9]]),
         array([[1., 0.7],
                [2., 0.9]]),
         array([], shape=(0, 2), dtype=float64)]
    """
    def __init__(self, num_classes):
        self._num_classes = num_classes
        self._pool = [[] for _ in range(num_classes)]

    def _type_cast(self, x):
        if type(x) == torch.Tensor:
            return x.numpy()
        elif type(x) == np.ndarray:
            return x
        else:
            raise TypeError('Unsupported data type {}'.format(type(x)))

    def fetch_class_data(self, cls):
        """
        Fetch ranked scores for a particular class

        Arguments:
            cls(int): class index

        Returns:
            list[array(N1,2), array(N2,2), array(N3,2)]: a list of three arrays, for
                positives, type-I negatives and type-II negatives respectively. Each
                Nx2 array contains sample index in the first column and ranked scores
                in the second column
        """
        assert type(cls) is int,\
                'Class index has to be an integer'
        assert cls >= 0 and cls < self._num_classes,\
                'Class index has to be between 0 and {NUM_CLASSES}'
        
        return self._pool[cls]

    def push(self, pred, labels):
        """
        Compute ranked score plots

        Arguments:
            pred(Tensor or ndarray): (N, C) prediction scores
            labels(Tensor or ndarray): (N, C) labels corresponding to the predictions
        """
        assert pred.shape == labels.shape,\
                'Predictions {} do not match the labels {} in size'.format(
                        pred.shape, labels.shape)
        assert labels.shape[1] == self._num_classes,\
                'Labels {} do not match the number of classes {}'.format(
                        labels.shape[1], self._num_classes)
        pred = self._type_cast(pred)
        labels = self._type_cast(labels)

        # find global positive samples
        gpos_inds = np.nonzero(np.sum(labels, 1))[0]
        # find type-II negative samples
        typeII_inds = np.nonzero(np.sum(labels, 1) == 0)[0]
        for cls in range(self._num_classes):
            pos_inds = np.nonzero(labels[:, cls])[0]
            typeI_inds = np.setdiff1d(gpos_inds, pos_inds)
            # append scores of positives
            samples = np.concatenate([
                pos_inds[:, None],
                pred[pos_inds, cls, None]
                ], 1)
            inds = np.argsort(samples[:, 1])
            self._pool[cls].append(samples[inds, :])
            # append scores of type-I negatives
            samples = np.concatenate([
                typeI_inds[:, None],
                pred[typeI_inds, cls, None]
                ], 1)
            inds = np.argsort(samples[:, 1])
            self._pool[cls].append(samples[inds, :])
            # append scores of type-II negatives
            samples = np.concatenate([
                typeII_inds[:, None],
                pred[typeII_inds, cls, None]
                ], 1)
            inds = np.argsort(samples[:, 1])
            self._pool[cls].append(samples[inds, :])

    def show(self, cls, order='default'):
        """
        Show the ranked score plot

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
            x1 = np.linspace(0, 1, len(self._pool[cls][0]))
            x2 = np.linspace(1, 0, len(self._pool[cls][1]))
            x3 = np.linspace(1, 0, len(self._pool[cls][2]))
            plt.xscale('log')
        else:
            raise ValueError('Unsupported ranking method \"{}\"'.format(order))

        # plot positives in green
        plt.plot(x1, self._pool[cls][0][:, 1], 'g', label='Positives')
        # plot type-I negatives in blue
        plt.plot(x2, self._pool[cls][1][:, 1], 'b', label='Type-I Negatives')
        # plot type-II negatives in red
        plt.plot(x3, self._pool[cls][2][:, 1], 'r', label='Type-II Negatives')

        plt.legend()
        plt.grid()
        plt.title('Ranked Score Plot for Class {}'.format(cls))
        plt.xlabel('Normalized sample indices')
        plt.ylabel('Classification scores')
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
                x1 = np.linspace(0, 1, len(self._pool[cls][0]))
                x2 = np.linspace(1, 0, len(self._pool[cls][1]))
                x3 = np.linspace(1, 0, len(self._pool[cls][2]))
                plt.xscale('log')
            else:
                raise ValueError('Unsupported ranking methods {}'.format(order))

            plt.plot(x1, self._pool[cls][0][:, 1], 'g', label='Positives')
            plt.plot(x2, self._pool[cls][1][:, 1], 'b', label='Type-I Negatives')
            plt.plot(x3, self._pool[cls][2][:, 1], 'r', label='Type-II Negatives')
            plt.legend()
            plt.grid()
            plt.title('Ranked Score Plot for Class {}'.format(cls))
            plt.xlabel('Normalized sample indices')
            plt.ylabel('Classification scores')
            plt.savefig(os.path.join(cache_dir, 'Class_{}.png'.format(cls)))
            plt.clf()


def compute_ranked_scores(
        net,
        labels,
        dataloader,
        device='cpu',
        cache_dir='./',
        multi_gpu=False,
        input_transform=lambda a: a,
        output_transform=lambda a: torch.cat([b for b in a], 0)):
        """
        Compute the ranked scores for three types of samples across all classes
            Positive
            Type-I Negative
            Type-II Negtative

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

        ranked_score = RankedScorePlot(labels.shape[1])
        ranked_score.push(output, labels)

        return ranked_score
