"""
Samplers used for torch.utils.data.DataLoader

Written by Frederic Zhang
Australian National Univeristy

Last updated in May 2019
"""

import torch
import numpy as np
import torch.utils.data

class StratifiedBatchSampler(torch.utils.data.Sampler):
    """
    Stratified sampler for a minibatch

    Given M strata/classes, form minibatches by taking an equal number of samples from N classes,
    where the N classes are taken sequentially from the entirety of M classes. When specified, a
    negative class can be appended in the batch

    Arguments:
        strata(list of Tensor or ndarray): (M,) strata indices
        num_strata_each(int): number of strata in each minibtach
        samples_per_stratum(int): number of samples taken from each stratum
        num_batch(int): number of minibatches to be sampled
        negative_pool(ndarray): the indices of negative samples
        num_negatives(int, optional): number of negative samples in the minibatch, default: 0
    """
    def __init__(self, 
            strata,
            num_strata_each,
            samples_per_stratum,
            num_batch,
            negative_pool=None,
            num_negatives=0):
        assert type(num_strata_each) is int,\
                'Number of strata in each minibatch should be an integer'
        assert num_strata_each <= len(strata),\
                'Number of strata in each minibatch cannot be larger than the total number of strata'
        assert type(samples_per_stratum) is int,\
                'Number of samples for each stratum has to be an integer'
        assert type(num_batch) is int,\
                'Number of minibatches has to be an interger'
        self._strata = strata
        self._num_strata_each = num_strata_each
        self._samples_per_stratum = samples_per_stratum
        self._num_batch = num_batch
        self._negative_pool = negative_pool
        self._num_negatives = num_negatives

        self._num_strata = len(strata)

    def __iter__(self):
        counter = 0
        for _ in range(self._num_batch):
            batch = []
            for i in range(self._num_strata_each):
                ind = (counter + i) % self._num_strata
                batch.append(self._strata[ind][\
                        torch.randint(high=len(self._strata[ind]), size=(1,)).item()].item())
            if self._negative_pool is not None:
                for i in range(self._num_negatives):
                    batch.append(self._negative_pool[\
                            torch.randint(high=len(self._negative_pool), size=(1,)).item()].item())
            yield batch
            counter += self._num_strata_each

    def __len__(self):
        return self._num_batch

class StratifiedSampler(torch.utils.data.Sampler):
    """
    Implementation of stratified sampling strategy

    Given M strata/classes and indices of samples from each stratum, take a specified number
    of samples from each stratum, and repeat a number of iterations

    Arguments:
        strata(list of Tensor or ndarray): (M,) strata indices
        num_iter(int): number of iterations to be sampled
        samples_per_stratum(Tensor or ndarray): (M,) number of samples taken from each stratum
    """
    def __init__(self, strata, num_iter, samples_per_stratum):
        assert len(strata) == len(samples_per_stratum),\
                'Number of strata {} not equal to the number of per-stratum samples specified'.\
                format(len(strata), len(samples_per_stratum))
        assert type(num_iter) is int,\
                'Number of iterations should be of type int'
        assert type(samples_per_stratum) in [np.ndarray, torch.Tensor],\
                'Samples per stratum should be a torch.Tensor or np.ndarray'
        self._strata = strata
        self._num_iter = num_iter
        self._samples_per_stratum = samples_per_stratum

    def __iter__(self):
        for _ in range(self._num_iter):
            for i, n in enumerate(self._samples_per_stratum):
                for _ in range(int(n.item())):
                    yield self._strata[i][\
                            torch.randint(high=len(self._strata[i]), size=(1,)).item()].item()

    def __len__(self):
        return self._num_iter * torch.sum(self._samples_per_stratum)


class MultiLabelStratifiedSampler(torch.utils.data.Sampler):
    """
    Stratified sampling strategy when samples belong to multiple classes
    
    Given M strata/classes and indices of samples from each stratum, when there could be
    potential overlap between  strata/classes, take a number of samples so that all classes
    have roughly the same number of samples. The number of samples desired for each class
    is specified in advance. 

    Algorithm:
        Prepare a counter for each of the stratum/class to record the number of samples. Take
        samples iteratively. When the number of samples a class has is no larger than the 
        current iteration number, take a sample for this class. Otherwise, skip the class. The
        number of iterations is equal to the specified number of samples needed

    Arguments:
        strata(list of Tensor or ndarray): (M,) strata indices
        labels(Tensor): (N, M) labels of all samples
        samples_per_class(int): number of samples taken per class
    """
    def __init__(self, strata, labels, samples_per_class):
        assert len(strata) == labels.shape[1],\
                'Number of strata {} not equal to number of classes {}'.\
                format(len(strata), labels.shape[1])
        assert type(labels) == torch.Tensor,\
                'Labels should a torch.Tensor or np.ndarray'
        assert type(samples_per_class) is int,\
                'Samples per class should be an integer'
        self._strata = strata
        self._labels = labels
        self._samples_per_class = samples_per_class
        # counter of samples for each class
        self._counter_per_class = torch.zeros(len(strata))
        # counter of total samples taken
        self._counter = None

    def __iter__(self):
        self._counter = 0
        self._counter_per_class = torch.zeros_like(self._counter_per_class)
        for i in range(self._samples_per_class):
            for j in range(len(self._strata)):
                if self._counter_per_class[j] > i:
                    continue
                else:
                    ind = self._strata[j][\
                            (torch.randint(high=len(self._strata[j]), size=(1,))).item()].item()
                    self._counter += 1
                    self._counter_per_class[torch.nonzero(self._labels[ind, :])[:, 0]] += 1
                    yield ind


    def __len__(self):
        if self._counter == None:
            raise NotImplementedError('Method __len__() not available before initializing a generator')
        else:
            return self._counter

    @property
    def counter(self):
        """Number of samples taken from each class"""
        return self._counter_per_class
