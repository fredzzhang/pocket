"""
Utilities for distributed training

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torch.distributed as dist

from .meters import NumericalMeter

class SyncedNumericalMeter(NumericalMeter):
    """
    Numerical meter synchronized across subprocesses
    """
    def __init__(self, maxlen=None):
        if not dist.is_available():
            raise AssertionError("Torch not compiled with distributed package")
        if not dist.is_initialized():
            raise AssertionError("Default process group has not been initialized")
        super().__init__(maxlen=maxlen)

    def append(self, x):
        """
        Append an elment

        Arguments:
            x(torch.Tensor, int or float)
        """
        if type(x) in NumericalMeter.VALID_TYPES:
            super().append(x)
        elif type(x) is torch.Tensor:
            super().append(x.item())
        else:
            raise TypeError('Unsupported data type {}'.format(x))

    def sum(self):
        """Compute the sum across all subprocesses"""
        sum_ = torch.sum(torch.as_tensor(self._deque))
        dist.barrier()
        dist.all_reduce(sum_)
        return sum_.item()

    def mean(self):
        """Compute the mean across all subprocesses"""
        mean_ = torch.mean(torch.as_tensor(self._deque))
        dist.barrier()
        dist.all_reduce(mean_)
        return mean_ / dist.get_world_size()

    def max(self):
        """Compute the max across all subprocesses"""
        max_ = torch.max(torch.as_tensor(self._deque))
        dist.barrier()
        dist.all_reduce(max_, op=dist.ReduceOp.MAX)
        return max_

    def min(self):
        """Compute the min across all subprocesses"""
        min_ = torch.min(torch.as_tensor(self._deque))
        dist.barrier()
        dist.all_reduce(min_, op=dist.ReduceOp.MIN)
        return min_