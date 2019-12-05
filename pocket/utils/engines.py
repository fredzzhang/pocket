"""
Learning engines under the PyTorch framework

Written by Frederic Zhang
Australian National University

Last updated in Dec. 2019
"""

import os
import copy
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .meters import NumericalMeter, HandyTimer
from .io import Log, TrainLog, load_pkl_from
from ..data import DataDict, ParallelOnlineBatchSampler, IndexSequentialSampler

class State:
    """
    Dict-based state class
    """
    def __init__(self):
        self._state = DataDict()

    def state_dict(self):
        """Return the state dict"""
        return self._state

    def load_state_dict(self, dict_in):
        """Load state from external dict"""
        for k in self._state:
            self._state[k] = dict_in[k]

    def fetch_state_key(self, key):
        """Return a specific key"""
        if key in self._state:
            return self._state[key]
        else:
            raise KeyError('Inexistent key {}'.format(key))

    def update_state_key(self, key, val):
        """Override a specific key in the state"""
        if key in self._state:
            assert type(val) == type(self._state[key]), \
                'Attemp to override state key \'{}\' of type {} by type {}'.format(key, type(self._state[key]), type(val))
            self._state[key] = val
        else:
            raise KeyError('Inexistent key {}'.format(key))

class LearningEngine(State):
    """
    Base class for learning engine
    """
    def __init__(self,
            net,
            criterion,
            trainloader,
            optim='SGD',
            optim_params={
                'lr': 0.001,
                'momentum': 0.9,
                'weight_decay': 5e-4
            },
            optim_state_dict=None,
            verbal=True,
            print_interval=100,
            cache_dir='./checkpoints'
            ):

        super(LearningEngine, self).__init__()

        self._device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')
        self._multigpu = torch.cuda.device_count() > 1
        self._criterion = criterion.to(self._device)
        self._trainloader = trainloader
        self._verbal = verbal
        self._print_interval = print_interval
        self._cache_dir = cache_dir

        # Set flags for GPU
        torch.backends.cudnn.benchmark = trainloader.pin_memory = torch.cuda.is_available()

        self._state.net = torch.nn.DataParallel(net).to(self._device) if self._multigpu \
            else net.to(self._device)
        self._state.optimizer = torch.optim.SGD(self._state.net.parameters(), **optim_params) if optim == 'SGD' \
            else torch.optim.Adam(self._state.net.parameters(), **optim_params)
        # Load optimzer state dict if provided
        if optim_state_dict is not None:
            self._state.optimizer.load_state_dict(optim_state_dict)
            # Relocate optimizer state to designated device
            for state in self._state.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self._state.device)
        self._state.epoch = 0
        self._state.iteration = 0

        self._state.running_loss = NumericalMeter()
        # Initialize timers
        self._state.t_data = NumericalMeter()
        self._state.t_iteration = HandyTimer()

    def __call__(self, n):
        # Train for a specified number of epochs
        for ep in range(n):
            self._on_start_epoch()

            timestamp = time.time()
            for batch in self._trainloader:
                self._state.input = batch[:-1]
                self._state.target = batch[-1]
                self._state.t_data.append(time.time() - timestamp)

                self._on_start_iteration()
                with self._state.t_iteration:
                    self._on_each_iteration()
                self._state.running_loss.append(self._state.loss.item())
                self._on_end_iteration()
                timestamp = time.time()
                
            self._on_end_epoch()

    def _on_start_epoch(self):
        pass

    def _on_end_epoch(self):
        self._state.epoch += 1
        self._save_checkpoint()

    def _on_start_iteration(self):
        self._state.input = [item.to(self._device) for item in self._state.input]
        self._state.target = self._state.target.to(self._device)

    def _on_end_iteration(self):
        self._state.iteration += 1
        if self._verbal and self._state.iteration % self._print_interval == 0:
            self._print_statistics()

    def _on_each_iteration(self):
        self._state.optimizer.zero_grad()
        self._state.output = self._state.net(*self._state.input)
        self._state.loss = self._criterion(self._state.output, self._state.target)
        self._state.loss.backward()
        self._state.optimizer.step()

    def _print_statistics(self):
        print('Epoch: [{}][{}]\t'
            'Iteration {:.3f}s ({:.3f}s)\t'
            'Dataloading {:.3f}s ({:.3f}s)\t'
            'Loss {:.4f} ({:.4f})'.format(
                self._state.epoch, self._state.iteration,
                self._state.t_iteration[-1], self._state.t_iteration.mean(),
                self._state.t_data[-1], self._state.t_data.mean(),
                self._state.loss.item(), self._state.running_loss.mean()))
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def _save_checkpoint(self):
        if not os.path.exists(self._cache_dir):
            os.mkdir(self._cache_dir)
        # Make a copy of the network and relocate to cpu
        net_copy = copy.deepcopy(self._state.net.module).cpu() if self._multigpu \
            else copy.deepcopy(self._state.net).cpu()
        # Make a copy of the optimizer and relocate to cpu
        optim_copy = copy.deepcopy(self._state.optimizer)
        for state in optim_copy.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        torch.save({
            'iteration': self._state.iteration,
            'epoch': self._state.epoch,
            'model_state_dict': net_copy.state_dict(),
            'optim_state_dict': optim_copy.state_dict()
            }, os.path.join(self._cache_dir, 'ckpt_{:05d}_{:02d}.pt'.\
                    format(self._state.iteration, self._state.epoch)))
