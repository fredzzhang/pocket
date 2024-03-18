"""
Principal component analysis

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning
"""

import torch

def pca(x, dim=2):
    """
    Perform PCA to for dimensionality reduction
    
    Arguments:
    ----------
    x: torch.Tensor
        Input data of size (N, K)
    dim: int
        Desired output dimension
        
    Returns:
    --------
    torch.Tensor
        Output data of size (N, dim)
    torch.Tensor
        Selected principal components of size (K, dim)
    """
    n = len(x)
    x_ = x - x.mean(0, keepdim=True)
    gram = x_.matmul(x_.T) / n
    e, v = torch.linalg.eig(gram)
    e = torch.real(e)
    v = torch.real(v)
    eigvec = (x_.T).matmul(v) / torch.sqrt(n * e.unsqueeze(0))

    order = torch.argsort(e, descending=True)
    pc = eigvec[:, order[:dim]]

    return x_.matmul(pc), pc