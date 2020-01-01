"""
Meters for the purpose of statistics tracking

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import time
import torch
import multiprocessing

from collections import deque

class Meter:
    """
    Base class
    """
    def __init__(self, maxlen=None):
        self._deque = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def __len__(self):
        return len(self._deque)

    def __iter__(self):
        return iter(self._deque)

    def __getitem__(self, i):
        return self._deque[i]

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += str([item for item in self._deque])
        reprstr += ', maxlen='
        reprstr += str(self._maxlen)
        reprstr += ')'
        return reprstr

    def reset(self):
        """Reset the meter"""
        self._deque.clear()

    def append(self, x):
        """Append an element"""
        self._deque.append(x)

    def sum(self):
        """Return the sum of all elements"""
        raise NotImplementedError

    def mean(self):
        """Return the mean"""
        raise NotImplementedError

    def max(self):
        """Return the minimum element"""
        raise NotImplementedError

    def min(self):
        """Return the maximum element"""
        raise NotImplementedError

    @property
    def items(self):
        """Return the content"""
        return [item for item in self._deque]

class NumericalMeter(Meter):
    """
    Meter class with numerals as elements
    """
    VALID_TYPES = [int, float]
    
    def __init__(self, maxlen=None):
        super().__init__(maxlen=maxlen)

    def append(self, x):
        if type(x) in self.VALID_TYPES:
            super().append(x)
        else:
            raise TypeError("Given element \'{}\' is not a numeral".format(x))

    def sum(self):
        return sum(self._deque)

    def mean(self):
        return sum(self._deque) / len(self._deque)

    def max(self):
        return max(self._deque)

    def min(self):
        return min(self._deque)

class HandyTimer(NumericalMeter):
    """
    A timer class that tracks a sequence of time
    """
    def __init__(self, maxlen=None):
        super().__init__(maxlen=maxlen)

    def __enter__(self):
        self._timestamp = time.time()

    def __exit__(self, type, value, traceback):
        self.append(time.time() - self._timestamp)

class AveragePrecisionMeter:
    """
    Meter to compute average precision

    Arguments:
        algorithm(str, optional): A choice between '11P' and 'AUC'
            '11P': 11-point interpolation algorithm prior to voc2010
            'INT': Interpolation algorithm with all points used in voc2010
            'AUC': Precisely as the area under precision-recall curve
        chunksize(int, optional): The approximate size the given iterable will be split
            into for each worker. Use -1 to make the argument adaptive to iterable size
            and number of workers
        output(tensor[N, K], optinoal): Network outputs with N examples and K classes
        target(tensor[N, K], optinoal): Binary labels
    """
    def __init__(self, algorithm="AUC", chunksize=-1, output=None, target=None):
        self._algorithm = algorithm
        self._chunksize = chunksize
        self._output = output.float() if output is not None \
            else torch.Tensor([])
        self._target = target.float() if target is not None \
            else torch.Tensor([])

        self._output_temp = [torch.Tensor([])]
        self._target_temp = [torch.Tensor([])]

    @staticmethod
    def compute_per_class_ap_as_auc(tuple_):
        """
        Arguments: 
            tuple_[(FloatTensor[N]), (FloatTensor[N])]: precision and recall
        Returns:
            ap(FloatTensor[1])
        """
        prec, rec = tuple_
        ap = 0
        for idx in range(prec.numel()):
            ap +=  prec[idx] * rec[idx] if idx == 0 \
                else 0.5 * (prec[idx] + prec[idx - 1]) * (rec[idx] - rec[idx - 1])
        return ap

    @staticmethod
    def compute_per_class_ap_with_interpolation(tuple_):
        """
        Arguments:
            tuple_[(FloatTensor[N]), (FloatTensor[N])]: precision and recall
        Returns:
            ap(FloatTensor[1])
        """
        prec, rec = tuple_
        ap = 0
        for idx in range(prec.numel()):
            # Precompute max for reuse
            max_ = prec[idx:].max()
            ap +=  max_ * rec[idx] if idx == 0 \
                else 0.5 * (max_ + torch.cat([prec[idx - 1], max_]).max()) * (rec[idx] - rec[idx - 1])
        return ap

    @staticmethod
    def compute_per_class_ap_with_11_point_interpolation(tuple_):
        """
        Arguments:
            tuple_[(FloatTensor[N]), (FloatTensor[N])]: precision and recall
        Returns:
            ap(FloatTensor[1])
        """
        prec, rec = tuple_
        ap = 0
        for t in torch.linspace(0, 1, 11):
            inds = torch.nonzero(rec >= t).squeeze()
            if inds.numel():
                ap += (prec[inds].max() / 11)
        return ap

    @classmethod            
    def compute_ap(cls, output, target, algorithm='AUC', chunksize=-1):
        """
        Compute AP precisely as the area under the precision-recall curve

        Arguments:
            output(FloatTensor[N, K])
            target(FloatTensor[N, K])
            chunksize(int, optional): The approximate size the given iterable will be split
                into for each worker. Use -1 to make the argument adaptive to iterable size
                and number of workers
        Returns:
            ap(FloatTensor[K])
        """
        prec, rec = cls.compute_precision_and_recall(output, target)
        ap = torch.zeros(output.shape[1])
        # Use the logic from pool._map_async to compute chunksize
        # https://github.com/python/cpython/blob/master/Lib/multiprocessing/pool.py
        # NOTE: Inappropriate chunksize will cause [Errno 24]Too many open files
        # Make changes with caution
        if chunksize == -1:
            chunksize, extra = divmod(
                    output.shape[1],
                    multiprocessing.cpu_count() * 4)
            if extra:
                chunksize += 1
       
        if algorithm == 'INT':
            algorithm_handle = cls.compute_per_class_ap_with_interpolation
        elif algorithm == '11P':
            algorithm_handle = cls.compute_per_class_ap_with_11_point_interpolation
        elif algorithm == 'AUC':
            algorithm_handle = cls.compute_per_class_ap_as_auc
        else:
            raise ValueError("Unknown algorithm option {}.".format(algorithm))

        with multiprocessing.Pool() as pool:
            for idx, output in enumerate(pool.imap(
                func=algorithm_handle,
                iterable=[(prec[:, k], rec[:, k]) for k in range(output.shape[1])],
                chunksize=chunksize
            )):
                ap[idx] = output
        
        return ap

    @staticmethod
    def compute_precision_and_recall(output, target, eps=1e-8):
        """
        Arguments:
            output(FloatTensor[N, K])
            target(FloatTensor[N, K])
            eps(float): A small constant to avoid division by zero
        Returns:
            prec(FloatTensor[N, K])
            rec(FloatTensor[N, K])
        """
        # Force data type
        output = output.float(); target = target.float()
        order = output.argsort(0, descending=True)
        tp = target[
            order,
            torch.ones_like(order) * torch.arange(output.shape[1])
        ]
        fp = 1 - tp
        tp = tp.cumsum(0)
        fp = fp.cumsum(0)

        prec = tp / (tp + fp)
        # NOTE: The insignificant constant could potentially result in 100%
        # recall being unreachable. Be cautious about its magnitude.
        rec = tp / (target.sum(0) + eps)
        return prec, rec

    def append(self, output, target):
        """
        Add new results to the meter

        Arguments:
            output(tensor[N, K]): Network output with N examples and K classes
            target(tensor[N, K]): Binary labels
        """
        if isinstance(output, torch.Tensor) and isinstance(target, torch.Tensor):
            self._output_temp.append(output.detach().cpu().float())
            self._target_temp.append(target.detach().cpu().float())
        else:
            raise TypeError("Arguments should both be torch.Tensor")

    def reset(self, keep_old=False):
        """
        Clear saved statistics

        Arguments:
            keep_tracked(bool): If True, clear only the newly collected statistics
                since last evaluation
        """
        if not keep_old:
            self._output = torch.Tensor([])
            self._target = torch.Tensor([])
        self._output_temp = [torch.Tensor([])]
        self._target_temp = [torch.Tensor([])]

    def eval(self):
        """
        Evaluate the average precision based on collected statistics

        Returns:
            torch.Tensor[K]: Average precisions for K classes
        """
        self._output = torch.cat([
            self._output,
            torch.cat(self._output_temp, 0)
        ], 0)
        self._target = torch.cat([
            self._target,
            torch.cat(self._target_temp, 0)
        ], 0)
        self.reset(keep_old=True)

        return self.compute_ap(output=self._output, target=self._target,
            algorithm=self._algorithm, chunksize=self._chunksize)
