"""
Relocate data

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch

def relocate_to_cpu(x):
    """Relocate data to cpu recursively"""
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, list):
        return [relocate_to_cpu(item) for item in x]
    elif isinstance(x, tuple):
        return (relocate_to_cpu(item) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_cpu(x[key])
        return x
    else:
        raise TypeError('Unsupported type of data {}'.format(type(x)))

def relocate_to_cuda(x, device=None, **kwargs):
    """
    Relocate data to CUDA recursively
    
    Arguments:
        x(Tensor, list, tuple or dict)
        device(torch.device or int)
        kwargs(dict): Refer to torch.Tensor.cuda() for keyworded arguments
    """
    if isinstance(x, torch.Tensor):
        return x.cuda(device, **kwargs)
    elif isinstance(x, list):
        return [relocate_to_cuda(item, device, **kwargs) for item in x]
    elif isinstance(x, tuple):
        return (relocate_to_cuda(item, device, **kwargs) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_cuda(x[key], device, **kwargs)
        return x
    else:
        raise TypeError('Unsupported type of data {}'.format(type(x)))

def relocate_to_device(x, device, **kwargs):
    """
    Relocate data to specified device recursively

    Arguments:
        x(Tensor, list, tuple or dict)
        device(torch.device, str or int)
        kwargs(dict): Refer to torch.Tensor.to() for keyworded arguments
    """
    if isinstance(x, torch.Tensor):
        return x.to(device, **kwargs)
    elif isinstance(x, list):
        return [relocate_to_device(item, device, **kwargs) for item in x]
    elif isinstance(x, tuple):
        return (relocate_to_device(item, device, **kwargs) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_device(x[key], device, **kwargs)
        return x
    else:
        raise TypeError('Unsupported type of data {}'.format(type(x)))