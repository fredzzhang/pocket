"""
Utilities related to tensor indexing

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch

from torch import Tensor, LongTensor

def intra_index(x: Tensor, y: Tensor, algorithm: str = 'broadcast') -> LongTensor:
    """
    Given 1D reference tensor x and query tensor y,
    find the index tensor z, such that `x[z] = y`

    Reference: https://discuss.pytorch.org/t/find-indices-of-one-tensor-in-another/84889

    Parameters:
    -----------
    x: Tensor
        (N,) Reference tensor
    y: Tensor
        (M,) Query tensor
    algorithm: str, default: `broadcast`
        `broadcast`: Much faster (recommended). Index for the last
            appearance of values in the query tensor is returned
        `loop`: Much slower but sligtly less memory consumption. Index
            for the first appearance is returned.

    Returns:
    --------
    z: LongTensor
        Index tensor 
    """
    if len(x.size()) != 1 or len(y.size()) != 1:
        raise ValueError(f"Both the reference tensor and query tensor should be 1D.")

    if algorithm == 'broadcast':
        z = -torch.ones_like(y).long()
        k, v = (y.unsqueeze(1) == x).nonzero(as_tuple=False).unbind(1)
        z[k] = v
        # Check if all elements in the query tensor have been retrieved
        err = torch.nonzero(z == -1).squeeze(1)
        if len(err):
            raise ValueError(f"The following values in the query tensor are not found: {y[err]}.")
        return z
    elif algorithm == 'loop':
        z = []
        for v in y:
            k = torch.nonzero(x == v).squeeze(1)
            if not len(k):
                raise ValueError(f"Value {v} in the query tensor is not found.")
            if len(k) > 1:
                k = k[0]
            z.append(k)
        return torch.as_tensor(z)
    else:
        raise ValueError(f"Unknown algorithm type {algorithm}")