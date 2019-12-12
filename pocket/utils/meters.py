"""
Meters for the purpose of statistics tracking

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import time
import torch

class Meter:
    """
    Base class
    """
    def __init__(self, x=list()):
        # NOTE: It is necessary to call .copy() as the default empty list
        # will be passed to EVERY instance of the class
        self._list = x.copy()

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += str(self._list)
        reprstr += ')'
        return reprstr

    def reset(self):
        """Reset meter"""
        self._list = []

    def append(self, x):
        """Append an element"""
        self._list.append(x)

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
        return self._list

class NumericalMeter(Meter):
    """
    Meter class with numerals as elements
    """
    VALID_TYPES = [int, float]
    
    def __init__(self, x=list()):
        for item in x:
            assert type(item) in self.VALID_TYPES, \
                "Given list contains non-numerical element {}".format(item)
        super().__init__(x)

    def append(self, x):
        if type(x) in self.VALID_TYPES:
            super().append(x)
        else:
            raise TypeError("Given element \'{}\' is not a numeral".format(x))

    def sum(self):
        return sum(self._list)

    def mean(self):
        return sum(self._list) / len(self._list)

    def max(self):
        return max(self._list)

    def min(self):
        return min(self._list)

class HandyTimer(NumericalMeter):
    """
    A timer class that tracks a sequence of time
    """
    def __init__(self):
        super().__init__()

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
        output(tensor[N, K], optinoal): Network outputs with N examples and K classes
        target(tensor[N, K], optinoal): Binary labels
    """
    def __init__(self, algorithm="11P", output=None, target=None):
        self._output = output.float() if output is not None \
            else torch.Tensor([])
        self._target = target.float() if target is not None \
            else torch.Tensor([])

        self._output_temp = [torch.Tensor([])]
        self._target_temp = [torch.Tensor([])]

        if algorithm == "11P":
            self._eval_alg = self.compute_ap_with_11_point_interpolation
        elif algorithm == 'INT':
            self._eval_alg = self.compute_ap_with_interpolation
        elif algorithm == "AUC":
            self._eval_alg = self.compute_ap_as_auc
        else:
            raise ValueError("Unknown algorithm option {}.".format(algorithm))

    @classmethod            
    def compute_ap_as_auc(cls, output, target):
        """
        Compute AP precisely as the area under the precision-recall curve

        Arguments:
            output(FloatTensor[N, K])
            target(FloatTensor[N, K])
        Returns:
            ap(FloatTensor[K])
        """
        prec, rec = cls.compute_precision_and_recall(output, target)
        ap = torch.zeros(output.shape[1])
        for k in range(output.shape[1]):
            for j in range(output.shape[0]):
                ap[k] +=  prec[j, k] * rec[j, k] if j == 0 \
                    else 0.5 * (prec[j, k] + prec[j-1, k]) * (rec[j, k] - rec[j-1, k])
        return ap

    @classmethod
    def compute_ap_with_interpolation(cls, output, target):
        """
        Compute AP with interpolation as per voc2010

        Arguments:
            output(FloatTensor[N, K])
            target(FloatTensor[N, K])
        Returns:
            ap(FloatTensor[K])
        """
        prec, rec = cls.compute_precision_and_recall(output, target)
        # TODO: Perform interpolation to make precision non-decreasing
        

        ap = torch.zeros(output.shape[1])
        for k in range(output.shape[1]):
            for j in range(output.shape[0]):
                ap[k] +=  prec[j, k] * rec[j, k] if j == 0 \
                    else 0.5 * (prec[j, k] + prec[j-1, k]) * (rec[j, k] - rec[j-1, k])
        return ap

    @classmethod
    def compute_ap_with_11_point_interpolation(cls, output, target):
        """
        Compute AP using 11-point interpolation algorithm as per voc
        challenge prior to voc2010

        Arguments:
            output(FloatTensor[N, K])
            target(FloatTensor[N, K])
        Returns:
            ap(FloatTensor[K])
        """
        prec, rec = cls.compute_precision_and_recall(output, target)
        ap = torch.zeros(output.shape[1])
        for k in range(output.shape[1]):
            for t in torch.linspace(0, 1, 11):
                inds = torch.nonzero(rec[:, k] >= t).squeeze()
                if inds.numel():
                    ap[k] += (prec[inds, k].max() / 11)
        return ap

    @staticmethod
    def compute_precision_and_recall(output, target, ep=1e-8):
        """
        Arguments:
            output(FloatTensor[N, K])
            target(FloatTensor[N, K])
            ep(float): A small constant to avoid division by zero
        Returns:
            prec(FloatTensor[N, K])
            rec(FloatTensor[N, K])
        """
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
        rec = tp / (target.sum(0) + ep)
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

        return self._eval_alg(self._output, self._target)