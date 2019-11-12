"""
Multilayer perceptron based on PyTorch framework

Written by Frederic Zhang
Australian National University

Last updated in Oct. 2019
"""

import torch
from torch import nn

class MLPwGN(nn.Module):
    """
    Multilayer perceptron with group normalization
    """
    def __init__(self, dimension, bias, num_groups):
        super(MLPwGN, self).__init__()
        self._dimension = dimension
        self._bias = bias
        self._num_layer = len(bias)
        self._num_groups = num_groups

        layers = [nn.Linear(dimension[0], dimension[1], bias[0])]
        for i in range(1, self._num_layer):
            layers.append(nn.GroupNorm(num_groups, dimension[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(
                dimension[i],
                dimension[i + 1],
                bias[i]
                ))
        self.layers = nn.Sequential(*layers)

    def __len__(self):
        """Return the number of linear layers"""
        return self._num_layer

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += repr(self._dimension)
        reprstr += ', '
        reprstr += repr(self._bias)
        reprstr += ', '
        reprstr += repr(self._num_groups)
        reprstr += ')'
        return reprstr

    def forward(self, x):
        # Group norm does not support 0-element inputs
        if  x.shape[0] == 0:
            return torch.zeros(0, self._dimension[-1])
        else:
            return self.layers(x)

    def freeze_bn(self):
        pass

    def activate_bn(self):
        pass

class MultiLayerPerceptron(nn.Module):
    """
    Multilayer perceptron class

    Arguments:
        dimension(list[int]): Dimension of layers in the MLP, starting from input
            layer. The length has to be at least 2
        bias(list[bool]): Whether to use bias term in a linear layer
        bathnorm(bool): If True, add batchnorm layers in between linear layers

    Example:

        >>> from pocket.models import MultiLayerPerceptron as MLP
        >>> net = MLP([5, 20, 10], [True, True])
        >>> net.layers
        Sequential(
          (0): Linear(in_features=5, out_features=20, bias=True)
          (1): ReLU()
          (2): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): Linear(in_features=20, out_features=10, bias=True)
        )
    """
    def __init__(self, dimension, bias, batchnorm=True):
        assert len(bias) + 1 == len(dimension),\
                'Contradictory arguments. Unable to infer the number of linear layers'
        super(MultiLayerPerceptron, self).__init__()
        self._dimension = dimension
        self._bias = bias
        self._num_layer = len(bias)
        self._batchnorm = batchnorm

        layers = [nn.Linear(dimension[0], dimension[1], bias[0])]
        for i in range(1, self._num_layer):
            if batchnorm:
                layers.append(nn.BatchNorm1d(dimension[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(
                dimension[i],
                dimension[i + 1],
                bias[i]
                ))
        self.layers = nn.Sequential(*layers)

    def __len__(self):
        """Return the number of linear layers"""
        return self._num_layer

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += repr(self._dimension)
        reprstr += ', '
        reprstr += repr(self._bias)
        reprstr += ', '
        reprstr += repr(self._batchnorm)
        reprstr += ')'
        return reprstr

    def freeze_bn(self):
        for layer in self.layers:
            if type(layer) == torch.nn.BatchNorm1d:
                layer.eval()
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

    def activate_bn(self):
        for layer in self.layers:
            if type(layer) == torch.nn.BatchNorm1d:
                layer.train()
                layer.weight.requires_grad = True
                layer.bias.requires_grad = True

    def forward(self, x):
        # In evaluation mode, BatchNorm module causes error when
        # the batch size of input data is zero
        if not self.training and x.shape[0] == 0:
            return torch.zeros(0, self._dimension[-1])
        else:
            return self.layers(x)
