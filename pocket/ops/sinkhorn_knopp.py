"""
Sinkhorn-Knopp normalisation

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch

def sinkhorn_knopp_norm2d(x, max_iter=1e3, tolerance=1e-3, eps=1e-6):
    """
    Sinkhorn-Knopp normalisation algorithm

    Adapted from:
        David Young (2020). Sinkhorn-Knopp algorithm for matrix normalisation,
        https://www.mathworks.com/matlabcentral/fileexchange
        MATLAB Central File Exchange.

    This implementation relaxes the constraint on input matrix to any non-negative
    matrices, including non-square ones and matrices with zero rows or columns. 
    
    Because zero rows and columns do not contribute to normalisation, they are
    ignored. In general, matrix of size MxN (already striped of zero rows and columns)
    is normalised such that the sum of each row equals 1/M and the sum of each column
    equals 1/N. This guarantees the matrix sums up to unity. Once converged, the matrix
    will be scaled by min{M, N}

    Arguments:
        x(Tensor[M, N] or np.ndarray[M, N] or list[list]): A non-negative
            2d array-like object that is convertable to a torch tensor.
        max_iter(int or float): The maximum number of iterations. Default: 1e3
        tolerance(float): Tolerance used to determine stopping condition. Default: 1e-3
        eps(float): Small constant to avoid division by zero
    Returns:
        Tensor[M, N]: Normalised matrix
        int: Number of iterations used
    """
    device = x.device if type(x) is torch.Tensor else None
    # Format input data
    x = torch.as_tensor(x,
        device=device,
        dtype=torch.float32)

    assert torch.all(x >= 0), "Given matrix contains negative entries"
    assert len(x.shape) == 2, "The dimensionality of given matrix is not 2"

    c_sum = x.sum(0)
    r_sum = x.sum(1)
    # Zero rows or columns do not contribute
    c_idx = (c_sum).nonzero().squeeze(1)
    r_idx = (r_sum).nonzero().squeeze(1)
    n_c = c_idx.numel()
    n_r = r_idx.numel()

    # The given matrix does not have non-zero elements
    if not n_c or not n_r:
        return x, 0

    rr, cc = torch.meshgrid(r_idx, c_idx)
    x_ = x[rr, cc]

    ratio = n_c / n_r
    # First iteration
    niter = 1
    # NOTE: Always normalise column sums to 1. Row sums will be normalised
    # accordinly. This implementation has better numerical stability when
    # there is significant divergence between the number of rows and columns. 
    c = 1 / (x_.sum(0) + eps)[None, :]
    r = 1 / (x_.mm(c.transpose(0, 1)) + eps) * ratio
    # Subsequent interations
    while niter < max_iter:
        niter += 1
        # Compute the column sums after row normalisation
        c_inv = r.transpose(0, 1).mm(x_)
        # Stop if column sums are within the tolerance of 1/N
        if (c_inv * c - 1).abs().max() < tolerance:
            break
        c = 1 / (c_inv + eps)
        r = 1 / (x_.mm(c.transpose(0, 1)) + eps) * ratio

    x_ = x_.mul_(r.mm(c))
    # Rescale the matrix if rows sums are larger than 1
    x[rr, cc] = x_ if ratio <= 1 else x_ / ratio

    return x, niter


class SinkhornKnoppNorm2d:
    """
    Sinkhorn-Knopp

    Refer to sinkhorn_knopp_norm2d for arguments
    """
    def __init__(self, max_iter=1e3, tolerance=1e-3, eps=1e-6):
        assert isinstance(max_iter, (int, float)), \
            "The maximum number of iterations must be an int or float, " \
            "not {}".format(type(max_iter))
        assert max_iter > 0, \
            "The maximum number of iterations must be positive, " \
            "not {}".format(max_iter)
        assert isinstance(tolerance, float), \
            "The tolerance must be a float, " \
            "not {}".format(type(tolerance))
        assert tolerance > 0 and tolerance < 1, \
            "The tolerance must be between 0 and 1, " \
            "not {}".format(tolerance)
        assert isinstance(eps, float), \
                "The small constant should be a float, " \
                "not {}".format(type(eps))

        self._max_iter = int(max_iter)
        self._tol = tolerance
        self._eps = eps
        self._iter = None

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += 'max_iter={}, '.format(self._max_iter)
        reprstr += 'tolerance={})'.format(self._tol)
        return reprstr

    @property
    def niter(self):
        """The iteration count in last call"""
        return self._iter

    @property
    def max_iter(self):
        """The maximum number of iterations"""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, n):
        self._max_iter = n

    @property
    def tolerance(self):
        """Tolerance used to determine stopping condition"""
        return self._tol

    def __call__(self, x):
        x, self._iter = sinkhorn_knopp_norm2d(x)
        return x
