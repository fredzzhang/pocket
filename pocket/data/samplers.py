"""
Samplers used for torch.utils.data.DataLoader

Written by Frederic Zhang
Australian National Univeristy

Last updated in Oct. 2019
"""

import os
import torch
import pickle
import datetime
import numpy as np
import torch.utils.data

class OnlineBatchSampler:
    """
    Online batch sampler that samples for subsequent batches by mixing highest scoring
    sampling from the previous batch and new samples

    Arguments:
        indices(list[N], np.ndarray[N] or torch.tensor[N]): Samples indices
        batch_size(int): Number of samples in a minibatch
        num_anchors(int): Number of samples taken from the previous batch
    """
    def __init__(self, indices, batch_size, num_anchors, randomize=False):
        if type(indices) == list:
            indices = np.asarray(indices)
        elif type(indices) == np.ndarray:
            assert len(indices.shape) == 1,\
                    'The given indices has to be 1-d array, not {}'.format(indices.shape)
            assert np.issubdtype(indices.dtype, np.integer),\
                    'Invalid data type {} for indices'.format(indices.dtype)
        elif type(indices) == torch.Tensor:
            indices = indices.numpy()
            assert len(indices.shape) == 1,\
                    'The given indices has to be 1-d Tensor, not {}'.format(indices.shape)
            assert np.issubdtype(indices.dtype, np.integer),\
                    'Invalid data type {} for indices'.format(indices.dtype)
        else:
            raise TypeError('Unsupported data type for given indices')

        if randomize:
            self._indices = indices[np.random.permutation(len(indices))]
        else:
            self._indices = indices
        self._batch_size = batch_size
        self._num_anchors = num_anchors
        self._anchors = np.array([])
        self._idx_ptr = 0

    @property
    def idx_ptr(self):
        return self._idx_ptr

    @property
    def anchors(self):
        return self._anchors

    @anchors.setter
    def anchors(self, x):
        assert type(x) == np.ndarray,\
                'Please use numpy.ndarray as anchor indices'
        assert x.shape == (self._num_anchors,),\
                'Anchor index array should have dimension ({},) not {}'.format(self._num_anchors, x.shape)
        self._anchors = x

    def next(self):

        if self._idx_ptr >= len(self._indices):
            raise StopIteration
        else:
            n_new_samples = self._batch_size - len(self._anchors)
            batch_indices = np.hstack([
                self._anchors,
                self._indices[self._idx_ptr: self._idx_ptr + n_new_samples]
                ])
            self._idx_ptr += n_new_samples

        return batch_indices.astype(np.int32)

class ParallelOnlineBatchSampler:
    r"""
    Multiple online batch samplers working alternately

    Arguments:
        indices(list[np.ndarray] or list[torch.tensor]): Sample indices
        batch_size(int): Number of samples in a batch
        num_anchors(int): Number of samples taken from the previous batch

    Example:

        >>> import numpy as np
        >>> from pocket.data import ParallelOnlineBatchSampler
        >>> # Generate indices
        >>> a = [np.array([1, 2, 3, 4, 5, 6, 7]),
                 np.array([8, 9, 10, 11, 12, 13]),
                 np.array([14, 15, 16, 17, 18])]
        >>> # Construct sampler, with batch size as 4 and 1 anchor
        >>> sampler = ParallelOnlineBatchSampler(a, 4, 1)
        >>> while(1):
        ...     try:
        ...         b, ptr = sampler.next()
        ...     except StopIteration:
        ...         break
        ...     print(b)
        ...     # Set the last element in a batch as the anchor in next batch
        ...     sampler.set_anchors(b[-1, None], ptr)
        ...
        [1 2 3 4]
        [8 9 10 11]
        [14 15 16 17]
        [4 5 6 7]
        [11 12 13]
        [17 18]

    """
    def __init__(self, indices, batch_size, num_anchors, shuffle=False):
        if shuffle:
            self._indices = [
                    seq[np.random.permutation(len(seq))] for seq in indices]
        else:
            self._indices = indices
        self._batch_size = batch_size
        self._num_anchors = num_anchors

        self._num_sampler = len(indices)
        self._anchors = [np.array([]) for _ in range(self._num_sampler)]

        self._sampler_ptr = 0
        self._active_samplers = [i for i in range(self._num_sampler)]
        self._idx_ptr = np.zeros(self._num_sampler, dtype=np.int32)

    @property
    def sampler_ptr(self):
        return self._active_samplers[self._sampler_ptr]

    def idx_ptr(self, i):
        return self._idx_ptr[i]

    def set_anchors(self, x, ptr):
        assert type(x) == np.ndarray,\
                'Please use numpy.ndarray as anchor indices'
        assert x.shape <= (self._num_anchors,),\
                'Number of anchors {} exceeds limit {}'.format(len(x), self._num_anchors)
        self._anchors[ptr] = x

    def next(self):

        if not len(self._active_samplers):
            raise StopIteration
        else:
            ptr = self._active_samplers[self._sampler_ptr]
            n_new_samples = self._batch_size - len(self._anchors[ptr])
            batch_indices = np.hstack([
                self._anchors[ptr],
                self._indices[ptr][
                    self._idx_ptr[ptr]: self._idx_ptr[ptr] + n_new_samples]
                ])
            self._idx_ptr[ptr] += n_new_samples

        # Deactivate sampler when exhausted
        if self._idx_ptr[ptr] >= len(self._indices[ptr]):
            self._active_samplers.pop(self._sampler_ptr)
            if self._sampler_ptr >= len(self._active_samplers):
                self._sampler_ptr = 0
        # Do not increment pointer when a sampler has exhausted
        elif len(self._active_samplers):
            self._sampler_ptr = (self._sampler_ptr + 1) % len(self._active_samplers)


        return batch_indices.astype(np.int32), ptr


class IndexSequentialSampler(torch.utils.data.Sampler):
    r"""
    Sequential sampler given a set of indices

    Arguments:
        indices(list[N], ndarray[N] or Tensor[N]): indices

    Example:

        >>> import torch
        >>> import numpy as np
        >>> from pocket.data import IndexSequentialSampler
        >>> # Construct a sampler from a list of indices
        >>> a = IndexSequentialSampler([1, 2, 3, 4])
        >>> for i in iter(a):
        ...     print(i)
        1
        2
        3
        4
        >>> # Construct a sampler from an array of indices
        >>> b = IndexSequentialSampler(np.array([1, 2, 3, 4]))
        >>> for i in iter(b):
        ...     print(i)
        1
        2
        3
        4
        >>> # Construct a sampler from a torch.Tensor
        >>> c = IndexSequentialSampler(torch.tensor([1, 2, 3, 4]))
        >>> for i in iter(c):
        ...     print(i)
        1
        2
        3
        4
    """
    def __init__(self, indices):
        if type(indices) == list:
            pass
        elif type(indices) == np.ndarray:
            assert len(indices.shape) == 1,\
                    'The given indices has to be 1-d array, not {}'.format(indices.shape)
            assert np.issubdtype(indices.dtype, np.integer),\
                    'Invalid data type {} for indices'.format(indices.dtype)
        elif type(indices) == torch.Tensor:
            indices = indices.numpy()
            assert len(indices.shape) == 1,\
                    'The given indices has to be 1-d Tensor, not {}'.format(indices.shape)
            assert np.issubdtype(indices.dtype, np.integer),\
                    'Invalid data type {} for indices'.format(indices.dtype)
        else:
            raise TypeError('Unsupported data type for given indices')
        self._indices = indices

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)

class StratifiedBatchSampler(torch.utils.data.Sampler):
    r"""
    Stratified sampler for a minibatch

    Given M strata/classes, form minibatches by taking an equal number of samples from N classes,
    where the N classes are taken sequentially from the entirety of M classes. When specified, samples
    from a negative class can be appended in the batch

    When sampling for a specific class, samples are taken randomly without replacement, util the
    class runs out of samples and gets renewed

    Arguments:
        strata(list[Tensor]): Strata indices
        num_strata_each(int): Number of strata in each minibtach
        samples_per_stratum(int): Number of samples taken from each stratum
        num_batch(int): Number of minibatches to be sampled
        negative_pool(ndarray, optional): The indices of negative samples, default: None
        num_negatives(int, optional): Number of negative samples in the minibatch, default: 0
        save_indices(bool, optional): If True, save the the sampled indices, default: False
        cache_dir(str, optional): Directory to save cached indices

    Example:
        
        >>> import torch
        >>> from pocket.data import StratifiedBatchSampler
        >>> # Generate strata indices
        >>> # Two strata are created in the following
        >>> strata = [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])]
        >>> # Generate negative sample indices
        >>> negatives = torch.tensor([6, 7, 8, 9])
        >>> # Construct stratified batch sampler
        >>> # Here each batch will take 2 samples ranomly from a selected stratum
        >>> # and 3 samples randomly from the negative pool. The selection of
        >>> # stratum is sequential, starting from the first one
        >>> a = StratifiedBatchSampler(strata, 1, 2, 5, negatives, 3)
        >>> for batch in a:
        ...     print(batch)
        [2, 1, 7, 8, 6]
        [4, 5, 9, 7, 8]
        [0, 1, 9, 6, 9]
        [3, 5, 7, 6, 8]
        [0, 2, 7, 8, 9]

    """
    def __init__(self, 
            strata,
            num_strata_each,
            samples_per_stratum,
            num_batch,
            negative_pool=None,
            num_negatives=0,
            save_indices=False,
            cache_dir='./'):
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
        self._save_indices = save_indices
        self._cache_dir = cache_dir

    def __iter__(self):
        counter = 0
        num_strata = len(self._strata)

        total_num_samples_per_stratum = self._num_batch * self._samples_per_stratum
        all_indices = torch.zeros(num_strata, total_num_samples_per_stratum, dtype=torch.int64)
        for i in range(num_strata):
            quot = total_num_samples_per_stratum // len(self._strata[i])
            rem = total_num_samples_per_stratum % len(self._strata[i])
            all_indices[i, :] = torch.cat([
                torch.cat([
                    self._strata[i][torch.randperm(len(self._strata[i]))] for _ in range(quot)
                    ]),
                self._strata[i][torch.randperm(len(self._strata[i]))[:rem]]
                ]) if quot != 0 else self._strata[i][torch.randperm(len(self._strata[i]))[:rem]]
        if self._negative_pool is not None:
            quot = self._num_batch * self._num_negatives // len(self._negative_pool)
            rem = self._num_batch * self._num_negatives % len(self._negative_pool)
            neg_indices = torch.cat([
                torch.cat([
                    self._negative_pool[torch.randperm(len(self._negative_pool))] for _ in range(quot)
                    ]),
                self._negative_pool[torch.randperm(len(self._negative_pool))[:rem]]
                ]) if quot != 0 else self._negative_pool[torch.randperm(len(self._negative_pool))[:rem]]

        all_batches = []
        for i in range(self._num_batch):
            batch = []
            for j in range(self._num_strata_each):
                stratum_id = (counter + j) % num_strata
                n = (counter + j) // num_strata
                stratum_samples = all_indices[
                        stratum_id, 
                        n * self._samples_per_stratum: (n + 1) * self._samples_per_stratum
                        ]
                for k in stratum_samples:
                    batch.append(k.item())
            if self._negative_pool is not None:
                neg_samples =  neg_indices[i * self._num_negatives: (i + 1) * self._num_negatives]
                for k in neg_samples:
                    batch.append(k.item())
            yield batch
            counter += self._num_strata_each

            if self._save_indices:
                all_batches.append(batch)
        if self._save_indices:
            with open(os.path.join(self._cache_dir,
                'Batch_{}.pkl'.format(str(datetime.datetime.now()))), 'wb') as f:
                pickle.dump(all_batches, f, pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return self._num_batch


# NOTE: THIS CLASS IS DEPRECATED
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

# NOTE: THIS CLASS IS DEPRECATED
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
