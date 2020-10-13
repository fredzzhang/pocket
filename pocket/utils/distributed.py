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
    Numerical meter synchronized across subprocesses. By default, it is assumed that
    NCCL is used as the communication backend. Communication amongst subprocesses can
    only be done with CUDA tensors, not CPU tensors. Make sure to intialise default 
    process group before instantiating the meter by

        torch.distributed.init_process_group(backbone="nccl", ...)
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

    def sum(self, local=False):
        """
        Arguments:
            local(bool): If True, return the local stats.
                Otherwise, aggregate over all subprocesses
        """
        if local:
            return torch.sum(torch.as_tensor(self._deque)).item()
        else:
            sum_ = torch.sum(torch.as_tensor(self._deque, device="cuda"))
            dist.barrier()
            dist.all_reduce(sum_)
            return sum_.item()

    def mean(self, local=False):
        """
        Arguments:
            local(bool): If True, return the local stats.
                Otherwise, aggregate over all subprocesses
        """
        if local:
            return torch.mean(torch.as_tensor(self._deque)).item()
        else:
            mean_ = torch.mean(torch.as_tensor(self._deque, device="cuda"))
            dist.barrier()
            dist.all_reduce(mean_)
            return mean_.item() / dist.get_world_size()

    def max(self, local=False):
        """
        Arguments:
            local(bool): If True, return the local stats.
                Otherwise, aggregate over all subprocesses
        """
        if local:
            return torch.max(torch.as_tensor(self._deque)).item()
        else:
            max_ = torch.max(torch.as_tensor(self._deque, device="cuda"))
            dist.barrier()
            dist.all_reduce(max_, op=dist.ReduceOp.MAX)
            return max_.item()

    def min(self, local=False):
        """
        Arguments:
            local(bool): If True, return the local stats.
                Otherwise, aggregate over all subprocesses
        """
        if local:
            return torch.min(torch.as_tensor(self._deque)).item()
        else:
            min_ = torch.min(torch.as_tensor(self._deque, device="cuda"))
            dist.barrier()
            dist.all_reduce(min_, op=dist.ReduceOp.MIN)
            return min_.item()