"""
Load/Initialize a network and prepare its
data formatter

Written by Frederic Zhang
Australian National University

Last updated in May 2019
"""

import torch
import numpy as np

class Interface:
    """Interface utilities for networks"""
    def __init__(self):
        pass

    @staticmethod
    def init_net(name, cfg):
        """Intialize a network"""
        raise NotImplementedError

    @classmethod
    def load_net(cls, name, cfg, path):
        """Intialize and load a network from checkpoint"""
        net = cls.init_net(name, cfg)
        net.load_state_dict(torch.load(path)['model_state_dict'])

        return net

    @classmethod
    def load_ckpt(cls, name, cfg, path):
        """
        Load a network and training progress
            (netword, optim_stat_dict, step, epoch)

        Assume the checkpoint file contains a dict with following keys
            step
            epoch
            model_state_dict
            optim_state_dict
        """
        net = cls.init_net(name, cfg)
        ckpt = torch.load(path)
        net.load_state_dict(ckpt['model_state_dict'])

        return net, ckpt['optim_state_dict'], ckpt['step'], ckpt['epoch']

    @staticmethod
    def get_formatter(name, mode):
        """Get the data formatter for a model given fetch mode of dataset"""
        raise NotImplementedError
