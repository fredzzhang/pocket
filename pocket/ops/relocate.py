"""
Relocate data

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch

from torch import Tensor
from typing import Optional, Union, List, Tuple, Dict, TypeVar

GenericTensor = TypeVar('GenericTensor', Tensor, List[Tensor], Tuple[Tensor], Dict[str, Tensor])

def relocate_to_cpu(x: GenericTensor) -> GenericTensor:
    """Relocate data to cpu recursively"""
    if isinstance(x, Tensor):
        return x.cpu()
    elif x is None:
        return x
    elif isinstance(x, list):
        return [relocate_to_cpu(item) for item in x]
    elif isinstance(x, tuple):
        return tuple(relocate_to_cpu(item) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_cpu(x[key])
        return x
    else:
        raise TypeError('Unsupported type of data {}'.format(type(x)))

def relocate_to_cuda(
        x: GenericTensor,
        device: Optional[Union[torch.device, int]] = None,
        **kwargs
    ) -> GenericTensor:
    """
    Relocate data to CUDA recursively
    
    Parameters:
    -----------
    x: Tensor, List[Tensor], Tuple[Tensor] or Dict[Tensor]
        Generic tensor data to be relocated
    device: torch.device or int
        Destination device
    kwargs: dict
        Refer to torch.Tensor.cuda() for keyworded arguments

    Returns:
    --------
    Tensor, List[Tensor], Tuple[Tensor] or Dict[Tensor]
        Relocated tensor data
    """
    if isinstance(x, torch.Tensor):
        return x.cuda(device, **kwargs)
    elif x is None:
        return x
    elif isinstance(x, list):
        return [relocate_to_cuda(item, device, **kwargs) for item in x]
    elif isinstance(x, tuple):
        return tuple(relocate_to_cuda(item, device, **kwargs) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_cuda(x[key], device, **kwargs)
        return x
    else:
        raise TypeError('Unsupported type of data {}'.format(type(x)))

def relocate_to_device(
        x: GenericTensor,
        device: Optional[Union[torch.device, str, int]] = None,
        **kwargs
    ) -> GenericTensor:
    """
    Relocate data to specified device recursively
    
    Parameters:
    -----------
    x: Tensor, List[Tensor], Tuple[Tensor] or Dict[Tensor]
        Generic tensor data to be relocated
    device: torch.device, str or int
        Destination device
    kwargs: dict
        Refer to torch.Tensor.to() for keyworded arguments

    Returns:
    --------
    Tensor, List[Tensor], Tuple[Tensor] or Dict[Tensor]
        Relocated tensor data
    """
    if isinstance(x, torch.Tensor):
        return x.to(device, **kwargs)
    elif x is None:
        return x
    elif isinstance(x, list):
        return [relocate_to_device(item, device, **kwargs) for item in x]
    elif isinstance(x, tuple):
        return tuple(relocate_to_device(item, device, **kwargs) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_device(x[key], device, **kwargs)
        return x
    else:
        raise TypeError('Unsupported type of data {}'.format(type(x)))