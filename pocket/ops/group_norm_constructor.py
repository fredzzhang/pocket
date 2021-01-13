"""
Group norm constructor

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from torch.nn import GroupNorm

class GroupNormConstructor:
    """
    A wrapper class to allow instantiating GroupNorm with one argument

    Arguments:
        num_groups(int or float): If an integer is given, the argument will be used
            as the number of groups. If a float is given, the argument will be used
            as percentage of channels in each group
        
        Refer to torch.nn.GroupNorm for kwargs
    """
    def __init__(self, num_groups, **kwargs):
        if type(num_groups) not in [int, float]:
            raise TypeError('Number of groups has to be either an integer or a float')
        self.num_groups = num_groups
        self.kwargs = kwargs

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += 'num_groups='
        reprstr += str(self.num_groups)
        for k in self.kwargs:
            reprstr += ', {}='.format(k)
            reprstr += str(self.kwargs[k])
        reprstr += ')'
        return reprstr

    def __call__(self, num_channels):
        if type(self.num_groups) is int:
            return GroupNorm(self.num_groups, num_channels, **self.kwargs)
        elif type(self.num_groups) is float:
            return GroupNorm(int(self.num_groups * num_channels), num_channels, **self.kwargs)
