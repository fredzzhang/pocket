"""
Meters for the purpose of statistics tracking

Written by Frederic Zhang
Australian National University

Last updated in Dec. 2019
"""

import time

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
    __valid__ = [int, float]
    
    def __init__(self, x=[]):
        for item in x:
            assert type(item) in self.__valid__, \
                'Given list contains non-numerical element {}'.format(item)
        super(NumericalMeter, self).__init__(x)

    def append(self, x):
        if type(x) in self.__valid__:
            super(NumericalMeter, self).append(x)
        else:
            raise TypeError('Given element \'{}\' is not a numeral'.format(x))

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