"""
Samplers

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import math
import copy
import torch
import pickle
import bisect
import datetime
import numpy as np
import torch.utils.data

from itertools import repeat, chain
from collections import defaultdict
from torch.utils.data.sampler import Sampler, BatchSampler

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

"""
Batch sampler that groups images by aspect ratio
https://github.com/pytorch/vision/blob/master/references/detection/group_by_aspect_ratio.py
"""

def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)

class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size

def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def create_aspect_ratio_groups(aspect_ratios, k=0, verbal=True):
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    if verbal:
        print("Using {} as bins for aspect ratio quantization".format(fbins))
        print("Count of instances per bin: {}".format(counts))
    return groups