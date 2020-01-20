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

def relocate_to_cuda(x, device):
    """
    Relocate data to CUDA recursively
    
    Arguments:
        x(Tensor, list, tuple or dict)
        device(torch.device or int)
    """
    device_id = torch.cuda._utils._get_device_index(device)
    if isinstance(x, torch.Tensor):
        return x.cuda(device_id)
    elif isinstance(x, list):
        return [relocate_to_cuda(item, device_id) for item in x]
    elif isinstance(x, tuple):
        return (relocate_to_cuda(item, device_id) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_cuda(x[key], device_id)
        return x
    else:
        raise TypeError('Unsupported type of data {}'.format(type(x)))

def relocate_to_device(x, device):
    """
    Relocate data to specified device recursively

    Arguments:
        x(Tensor, list, tuple or dict)
        device(torch.device, str or int)
    """
    device = torch.device(device)
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [relocate_to_device(item, device) for item in x]
    elif isinstance(x, tuple):
        return (relocate_to_device(item, device) for item in x)
    elif isinstance(x, dict):
        for key in x:
            x[key] = relocate_to_device(x[key], device)
        return x
    else:
        raise TypeError('Unsupported type of data {}'.format(type(x)))