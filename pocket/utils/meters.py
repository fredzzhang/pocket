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
    def __init__(self, x=[]):
        self._list = x

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

class NumericalMeter(Meter):
    """
    Meter class with numerals as elements
    """
    VALID_TYPES = [int, float]
    
    def __init__(self, x=[]):
        for item in x:
            assert type(item) in self.VALID_TYPES, \
                "Given list contains non-numerical element {}".format(item)
        super(NumericalMeter, self).__init__(x)

    def append(self, x):
        if type(x) in self.VALID_TYPES:
            super(NumericalMeter, self).append(x)
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
        super(HandyTimer, self).__init__()

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
        self._output = output if output is not None \
            else torch.Tensor([])
        self._target = target if target is not None \
            else torch.Tensor([])

        self._output_temp = []
        self._target_temp = []

        if algorithm == "11P":
            self._eval_alg = self.compute_ap_with_11_point_interpolation
        elif algorithm == 'INT':
            self._eval_alg = self.compute_ap_with_interpolation
        elif algorithm == "AUC":
            self._eval_alg = self.compute_ap_as_auc
        else:
            raise ValueError("Unknown algorithm option {}.".format(algorithm))

    @staticmethod            
    def compute_ap_as_auc(output, target):
        """
        Arguments:
            output(tensor[N, K])
            target(tensor[N, K])
        Returns:
            tensor[K]
        """
        raise NotImplementedError
    
    @staticmethod
    def compute_ap_with_interpolation(output, target):
        """
        Arguments:
            output(tensor[N, K])
            target(tensor[N, K])
        Returns:
            tensor[K]
        """
        raise NotImplementedError

    @staticmethod
    def compute_ap_with_11_point_interpolation(output, target):
        """
        Arguments:
            output(tensor[N, K])
            target(tensor[N, K])
        Returns:
            tensor[K]
        """
        raise NotImplementedError


    def append(self, output, target):
        """
        Add new results to the meter

        Arguments:
            output(tensor[N, K]): Network output with N examples and K classes
            target(tensor[N, K]): Binary labels
        """
        if isinstance(output, torch.Tensor) and isinstance(target, torch.Tensor):
            self._output_temp.append(output.detach().cpu())
            self._target_temp.append(target.detach().cpu())
        else:
            raise TypeError("Arguments should both be torch.Tensor")

    def reset(self, keep_tracked=False):
        """
        Clear saved statistics
        """
        if not keep_tracked:
            self._output = torch.Tensor([])
            self._target = torch.Tensor([])
        self._output_temp = []
        self._target_temp = []

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
        self.reset(keep_tracked=True)

        return self._eval_alg(self._output, self._target)