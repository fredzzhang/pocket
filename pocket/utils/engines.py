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

from .meters import NumericalMeter, HandyTimer
from ..data import DataDict

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
    r"""
    Base class for learning engine

    Arguments:

    [REQUIRED ARGS]
        net(Module): The network to be trained
        criterion(Module): Loss function
        train_loader(iterable): Dataloader for training set, with batch input in the
            format [INPUT_1, ..., INPUT_N, LABELS]

    [OPTIONAL ARGS]
        optim(str): Optimizer to be used. Choose between 'SGD' and 'Adam'
        optim_params(dict): Parameters for the selected optimizer
        optim_state_dict(dict): Optimizer state dict to be loaded
        lr_scheduler(bool): If True, use MultiStepLR as the learning rate scheduler
        lr_sched_params(dict): Parameters for the learning rate scheduler
        verbal(bool): If True, print statistics every fixed interval
        print_interval(int): Number of iterations to print statistics
        cache_dir(str): Directory to save checkpoints
    """
    def __init__(self,
            net,
            criterion,
            train_loader,
            optim='SGD',
            optim_params={
                'lr': 0.001,
                'momentum': 0.9,
                'weight_decay': 5e-4
            },
            optim_state_dict=None,
            lr_scheduler=False,
            lr_sched_params={
                'milestones': [50,100],
                'gamma': 0.1
            },
            verbal=True,
            print_interval=100,
            cache_dir='./checkpoints'
            ):

        super(LearningEngine, self).__init__()

        self._device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')
        self._multigpu = torch.cuda.device_count() > 1
        self._criterion = criterion.to(self._device)
        self._train_loader = train_loader
        self._verbal = verbal
        self._print_interval = print_interval
        self._cache_dir = cache_dir

        # Set flags for GPU
        torch.backends.cudnn.benchmark = train_loader.pin_memory = torch.cuda.is_available()

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

        # Initialize learning rate scheduler
        self._lr_scheduler = None if not lr_scheduler \
            else torch.optim.lr_scheduler.MultiStepLR(self._state.optimizer, **lr_sched_params)

        self._state.running_loss = NumericalMeter()
        # Initialize timers
        self._state.t_data = NumericalMeter()
        self._state.t_iteration = HandyTimer()

    def __call__(self, n):
        # Train for a specified number of epochs
        for _ in range(n):
            self._on_start_epoch()
            timestamp = time.time()
            for batch in self._train_loader:
                self._state.input = batch[:-1]
                self._state.target = batch[-1]
                self._state.t_data.append(time.time() - timestamp)

                self._on_start_iteration()
                # Force network mode
                self._state.net.train()
                with self._state.t_iteration:
                    self._on_each_iteration()
                self._state.running_loss.append(self._state.loss.item())
                self._on_end_iteration()
                timestamp = time.time()
                
            self._on_end_epoch()

    def _on_start_epoch(self):
        self._state.epoch += 1

    def _on_end_epoch(self):
        self._save_checkpoint()
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    def _on_start_iteration(self):
        self._state.iteration += 1
        self._state.input = [item.to(self._device) for item in self._state.input]
        self._state.target = self._state.target.to(self._device)

    def _on_end_iteration(self):
        if self._verbal and self._state.iteration % self._print_interval == 0:
            self._print_statistics()

    def _on_each_iteration(self):
        self._state.optimizer.zero_grad()
        self._state.output = self._state.net(*self._state.input)
        self._state.loss = self._criterion(self._state.output, self._state.target)
        self._state.loss.backward()
        self._state.optimizer.step()

    def _print_statistics(self):
        print('[Ep.][Iter.]: [{}][{}] | '
                'Loss: {:.4f} | '
                'Time[Data][Iter.]: [{:.4f}s][{:.4f}s]'.format(
                    self._state.epoch, self._state.iteration,
                    self._state.running_loss.mean(),
                    self._state.t_iteration.sum(),
                    self._state.t_data.sum())
            )
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

class MultiClassClassificationEngine(LearningEngine):
    r"""
    Learning engine for multi-class classification problems

    Arguments:

    [REQUIRED ARGS]
        net(Module): The network to be trained
        criterion(Module): Loss function
        train_loader(iterable): Dataloader for training set, with batch input in the
            format [INPUT_1, ..., INPUT_N, LABELS]

    [OPTIONAL ARGS]
        val_loader(iterable): Dataloader for validation set, with batch input in the
            format [INPUT_1, ..., INPUT_N, LABELS]
        optim(str): Optimizer to be used. Choose between 'SGD' and 'Adam'
        optim_params(dict): Parameters for the selected optimizer
        optim_state_dict(dict): Optimizer state dict to be loaded
        lr_scheduler(bool): If True, use MultiStepLR as the learning rate scheduler
        lr_sched_params(dict): Parameters for the learning rate scheduler
        verbal(bool): If True, print statistics every fixed interval
        print_interval(int): Number of iterations to print statistics
        cache_dir(str): Directory to save checkpoints

    Example:

        >>> # An example on MNIST handwritten digits recognition
        >>> import torch
        >>> from torchvision import datasets, transforms
        >>> from pocket.models import LeNet
        >>> from pocket.utils import MultiClassClassificationEngine
        >>> # Fix random seed
        >>> torch.manual_seed(0)
        >>> # Initialize network
        >>> net = LeNet()
        >>> # Initialize loss function
        >>> criterion = torch.nn.CrossEntropyLoss()
        >>> # Prepare dataset
        >>> train_loader = torch.utils.data.DataLoader(
        ...     datasets.MNIST('./data', train=True, download=True,
        ...         transform=transforms.Compose([
        ...             transforms.ToTensor(),
        ...             transforms.Normalize((0.1307,), (0.3081,))])
        ...         ),
        ...     batch_size=128, shuffle=True)
        >>> test_loader = torch.utils.data.DataLoader(
        ...     datasets.MNIST('./data', train=False, 
        ...             transform=transforms.Compose([
        ...                 transforms.ToTensor(),
        ...                 transforms.Normalize((0.1307,), (0.3081,))])
        ...         ),
        ...     batch_size=100, shuffle=False)
        >>> # Intialize learning engine and start training
        >>> engine = MultiClassClassificationEngine(net, criterion, train_loader,
        ...     val_loader=test_loader)
        >>> # Train the network for one epoch with default optimizer option
        >>> # Checkpoints will be saved under ./checkpoints by default, containing 
        >>> # saved model parameters, optimizer statistics and progress
        >>> engine(1)

        => Validation
        Epoch: 0 | Acc.: 0.1008[1008/10000] | Loss: 2.3036 | Time: 5.24s

        [Ep.][Iter.]: [1][100] | Loss: 1.1665 | Time[Data][Iter.]: [466.5847s][466.5847s]
        [Ep.][Iter.]: [1][200] | Loss: 2.2773 | Time[Data][Iter.]: [0.2861s][2.6689s]
        [Ep.][Iter.]: [1][300] | Loss: 2.2289 | Time[Data][Iter.]: [0.2581s][2.5305s]
        [Ep.][Iter.]: [1][400] | Loss: 2.0143 | Time[Data][Iter.]: [0.2986s][2.6412s]

        => Training
        Epoch: 1 | Acc.: 0.3181[19087/60000]
        => Validation
        Epoch: 1 | Acc.: 0.7950[7950/10000] | Loss: 1.0872 | Time: 2.18s
    """
    def __init__(self,
            net,
            criterion,
            train_loader,
            val_loader=None,
            **kwargs):

        super(MultiClassClassificationEngine, self).__init__(net, criterion, train_loader, **kwargs)
        val_loader.pin_memory = torch.cuda.is_available()
        self._val_loader = val_loader
           
    def _validate(self):
        self._state.net.eval()
        correct = 0
        total = 0
        running_loss = NumericalMeter()
        timestamp = time.time()
        for batch in self._val_loader:
            batch = [item.to(self._device) for item in batch]
            with torch.no_grad():
                output = self._state.net(*batch[:-1])
            loss = self._criterion(output, batch[-1])
            running_loss.append(loss.item())
            pred = torch.argmax(output, 1)
            correct += torch.eq(pred, batch[-1]).sum().item()
            total += len(pred)
        elapsed = time.time() - timestamp

        print('=> Validation\n'
            'Epoch: {} | Acc.: {:.4f}[{}/{}] | Loss: {:.4f} | Time: {:.2f}s\n'.format(
                self._state.epoch, correct / total, correct, total,
                running_loss.mean(), elapsed
            ))

    def _on_start_epoch(self):
        if self._state.epoch == 0 and self._val_loader is not None:
            self._validate()
        super(MultiClassClassificationEngine, self)._on_start_epoch()
        self._state.correct = 0
        self._state.total = 0
 
    def _on_end_epoch(self):
        super(MultiClassClassificationEngine, self)._on_end_epoch()
        print('\n=> Training\n'
            'Epoch: {} | Acc.: {:.4f}[{}/{}]'.format(
                self._state.epoch,
                self._state.correct / self._state.total, self._state.correct, self._state.total
            ))
        if self._val_loader is not None:
            self._validate()

    def _on_end_iteration(self):
        super(MultiClassClassificationEngine, self)._on_end_iteration()
        pred = torch.argmax(self._state.output, 1)
        self._state.correct += torch.eq(pred, self._state.target).sum().item()
        self._state.total += len(pred)
