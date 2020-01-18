"""
Multilayer perceptron based on PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
    """
    Multilayer perceptron class

    Arguments:
        dimension(list[int]): Dimension of layers in the MLP, starting from input
            layer. The length has to be at least 2
        bias(bool or list[bool], optional): If True, use bias terms in linear layers. 
            If given a bool variable, apply it to all linear layers, otherwise apply 
            to individual linear layer in order
        use_norm(bool, optional): If True, use normalization layer before activation (ReLu)
        norm_layer(callable, optional): Normalization layer to be used

    Example:

        >>> from pocket.models import MultiLayerPerceptron as MLP
        >>> net = MLP([5, 20, 10])
        >>> net.layers
        Sequential(
            (0): Linear(in_features=5, out_features=20, bias=True)
            (1): ReLU()
            (2): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Linear(in_features=20, out_features=10, bias=True)
        )
        >>> from pocket.ops import GroupNormConstructor
        >>> norm = GroupNormConstructor(32)
        >>> net = MLP([1024, 1024, 100], bias=[True, False], norm_layer=norm)
        >>> net.layers
        Sequential(
            (0): Linear(in_features=1024, out_features=1024, bias=True)
            (1): GroupNorm(32, 1024, eps=1e-05, affine=True)
            (2): ReLU()
            (3): Linear(in_features=1024, out_features=100, bias=False)
        )
    """
    def __init__(self, dimension, bias=True, use_norm=True, norm_layer=None):
        super(MultiLayerPerceptron, self).__init__()
        self._dimension = dimension
        self._num_layer = len(dimension) - 1
        self._bias = bias if type(bias) is list \
            else [bias for _ in range(self._num_layer)]
        self._use_norm = use_norm
        self._norm_layer = nn.BatchNorm1d if norm_layer is None \
            else norm_layer

        if not use_norm and norm_layer is not None:
            print('WARNING: The passed normalization layer is not used')

        layers = [nn.Linear(dimension[0], dimension[1], self._bias[0])]
        for i in range(1, self._num_layer):
            if use_norm:
                layers.append(self._norm_layer(dimension[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(
                dimension[i],
                dimension[i + 1],
                self._bias[i]
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
        reprstr += ', use_norm='
        reprstr += repr(self._use_norm)
        reprstr += ', norm_layer='
        reprstr += repr(self._norm_layer)
        reprstr += ')'
        return reprstr

    def forward(self, x):
        # In evaluation mode, BatchNorm module causes error when
        # the batch size of input data is zero
        if not self.training and x.shape[0] == 0:
            return torch.zeros(0, self._dimension[-1])
        else:
            return self.layers(x)
