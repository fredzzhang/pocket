"""
Sinkhorn-Knopp normalisation

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import warnings

class SinkhornKnoppNorm:
    """
    Sinkhorn-Knopp algorithm

    Adapted from:
        David Young (2020). Sinkhorn-Knopp algorithm for matrix normalisation,
        https://www.mathworks.com/matlabcentral/fileexchange
        MATLAB Central File Exchange.

    This implementation relaxes the constraint on input matrix to any non-negative
    matrices, including non-square ones, and performs row and column normalisation 
    iteratively. The algorithm stops when either the maximum number of iterations 
    is reached or the sum of each row and column is within a tolerance of 1.

    Arguments:
        max_iter(int or float): The maximum number of iterations. Default: 1e3
        tolerance(float): Tolerance used to determine stopping condition. Default: 1e-3
    """
    def __init__(self, max_iter=1e3, tolerance=1e-3):
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

        self._max_iter = int(max_iter)
        self._tol = tolerance
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

    def __call__(self, x, eps=1e-6):
        """
        Normalise a given matrix

        Arguments:
            x(Tensor[M, N] or np.ndarray[M, N] or list[list]): A non-negative
                2d array-like object that is convertable to a torch tensor.
            eps(float): Small constant to avoid division by zero

        Returns:
            Tensor[M, N]
        """
        device = x.device if type(x) is torch.Tensor else None
        # Format input data
        x = torch.as_tensor(x,
            device=device,
            dtype=torch.float32)

        assert torch.all(x >= 0), "Given matrix contains negative entries"
        assert x.ndim == 2, "The dimensionality of given matrix is not 2"

        if x.shape[0] != x.shape[1]:
            warnings.warn("The given matrix is not a square matrix. "
                "The algorithm may not converge", UserWarning)

        c_sum = x.sum(0)
        r_sum = x.sum(1)

        if not torch.all(c_sum > 0) or not torch.all(r_sum > 0):
            warnings.warn("The given matrix contains rows or columns of zeros. "
                "The algorithm may not converge", UserWarning)

        # First iteration
        niter = 1
        # Column sums are kept as a row vector and row sums a column vector
        c = 1 / (c_sum + eps)[None, :]
        r = 1 / (x.mm(c.T) + eps)
        # Subsequent interations
        while niter < self._max_iter:
            niter += 1
            # Compute the column sums after row normalisation
            c_inv = r.T.mm(x)
            # Stop if column sums are within the tolerance of 1
            if (c_inv * c - 1).abs().max() < self._tol:
                break
            c = 1 / (c_inv + eps)
            r = 1 / (x.mm(c.T) + eps)

        # Update iteration counter
        self._iter = niter

        return x * r.mm(c)