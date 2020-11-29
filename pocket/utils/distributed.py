"""
Utilities for distributed training

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import pickle
import torch.distributed as dist

from typing import Optional, Union, Any, List
from .meters import NumericalMeter

class SyncedNumericalMeter(NumericalMeter):
    """
    Numerical meter synchronized across subprocesses. By default, it is assumed that
    NCCL is used as the communication backend. Communication amongst subprocesses can
    only be done with CUDA tensors, not CPU tensors. Make sure to intialise default 
    process group before instantiating the meter by

        torch.distributed.init_process_group(backbone="nccl", ...)
    """
    def __init__(self, maxlen: Optional[int] = None) -> None:
        if not dist.is_available():
            raise AssertionError("Torch not compiled with distributed package")
        if not dist.is_initialized():
            raise AssertionError("Default process group has not been initialized")
        super().__init__(maxlen=maxlen)

    def append(self, x: Union[int, float]) -> None:
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

    def sum(self, local: bool = False) -> Union[int, float]:
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

    def mean(self, local: bool = False) -> float:
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

    def max(self, local: bool = False) -> Union[int, float]:
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

    def min(self, local: bool = False) -> Union[int, float]:
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

def all_gather(data: Any) -> List[Any]:
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    https://github.com/pytorch/vision/blob/master/references/detection/utils.py

    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list