"""
Distributed learning engines based on torch.distributed

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import time
import torch
import torch.distributed as dist

from torch.nn import Module
from typing import Callable, Iterable, Optional

from ..ops import relocate_to_cuda
from ..utils import SyncedNumericalMeter

from .engines import State

class DistributedLearningEngine(State):
    r"""
    Base class for distributed learning engine

    Arguments:

    [REQUIRED ARGS]
        net(Module): The network to be trained
        criterion(callable): Loss function
        train_loader(iterable): Dataloader for training set, with batch input in the
            format [INPUT_1, ..., INPUT_N, LABELS]. Each element should take one of 
            the following forms: Tensor, list[Tensor], dict[Tensor]

    [OPTIONAL ARGS]
        device(int or torch.device): CUDA device to be used for training
        optim(str): Optimizer to be used. Choose between 'SGD' and 'Adam'
        optim_params(dict): Parameters for the selected optimizer
        optim_state_dict(dict): Optimizer state dict to be loaded
        lr_scheduler(bool): If True, use MultiStepLR as the learning rate scheduler
        lr_sched_params(dict): Parameters for the learning rate scheduler
        verbal(bool): If True, print statistics every fixed interval
        print_interval(int): Number of iterations to print statistics
        cache_dir(str): Directory to save checkpoints
        find_unused_parameters(bool): torch.nn.parallel.DistributedDataParallel
    """
    def __init__(self,
            net: Module, criterion: Callable, train_loader: Iterable,
            device: Optional[int] = None, optim: str = 'SGD',
            optim_params: Optional[dict] = None, optim_state_dict: Optional[dict] = None,
            lr_scheduler: bool = False, lr_sched_params: Optional[dict] = None,
            verbal: bool = True, print_interval: int = 100, use_amp: bool = True,
            find_unused_parameters: bool = False, cache_dir: str = './checkpoints'
        ):

        super().__init__()
        if not dist.is_available():
            raise AssertionError("Torch not compiled with distributed package")
        if not dist.is_initialized():
            raise AssertionError("Default process group has not been initialized")

        self._dawn = time.time()

        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._device = torch.device(device) if device is not None else torch.device(self._rank)
        # Set the default device
        # NOTE Removing this line causes non-master subprocesses stuck at data loading
        torch.cuda.set_device(self._device)

        self._criterion = criterion if not isinstance(criterion, torch.nn.Module) \
            else criterion.cuda()
        self._train_loader = train_loader
        self._verbal = verbal
        self._print_interval = print_interval
        self._use_amp = use_amp
        self._cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Relocate model to designated device
        net.cuda()

        # Initialize optimizer
        net_params = [p for p in net.parameters() if p.requires_grad]
        if optim_params is None:
            optim_params = {
                    'lr': 0.001,
                    'momentum': 0.9,
                    'weight_decay': 5e-4
            } if optim == 'SGD' else {'lr': 0.001, 'weight_decay': 5e-4}
        self._state.optimizer = eval(f'torch.optim.{optim}')(net_params, **optim_params)
        # Load optimzer state dict if provided
        if optim_state_dict is not None:
            self._state.optimizer.load_state_dict(optim_state_dict)
            # Relocate optimizer state to designated device
            for state in self._state.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        self._state.net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[self._device],
            find_unused_parameters=find_unused_parameters
        )

        # Initialise gradient scaler
        self._state.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self._state.epoch = 0
        self._state.iteration = 0

        # Initialize learning rate scheduler
        lr_sched_params = {
                'milestones': [50,100],
                'gamma': 0.1
            } if lr_sched_params is None else lr_sched_params
        self._state.lr_scheduler = None if not lr_scheduler \
            else torch.optim.lr_scheduler.MultiStepLR(self._state.optimizer, **lr_sched_params)

        self._state.running_loss = SyncedNumericalMeter(maxlen=print_interval)
        # Initialize timers
        self._state.t_data = SyncedNumericalMeter(maxlen=print_interval)
        self._state.t_iteration = SyncedNumericalMeter(maxlen=print_interval)

    def __call__(self, n: int) -> None:
        self.epochs = n
        # Train for a specified number of epochs
        self._on_start()
        for _ in range(n):
            self._on_start_epoch()
            timestamp = time.time()
            for batch in self._train_loader:
                self._state.inputs = batch[:-1]
                self._state.targets = batch[-1]
                self._on_start_iteration()
                self._state.t_data.append(time.time() - timestamp)

                self._on_each_iteration()
                self._state.running_loss.append(self._state.loss.item())
                self._on_end_iteration()
                self._state.t_iteration.append(time.time() - timestamp)
                timestamp = time.time()
                
            self._on_end_epoch()
        self._on_end()

    def _on_start(self):
        pass

    def _on_end(self):
        pass

    def _on_start_epoch(self):
        self._state.epoch += 1
        # Force network mode
        self._state.net.train()
        # Update random seeds for sampler
        self._train_loader.sampler.set_epoch(self._state.epoch)

    def _on_end_epoch(self):
        # Save checkpoint in the master process
        if self._rank == 0:
            self.save_checkpoint()
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()

    def _on_start_iteration(self):
        self._state.iteration += 1
        self._state.inputs = relocate_to_cuda(self._state.inputs, non_blocking=True)
        self._state.targets = relocate_to_cuda(self._state.targets, non_blocking=True)

    def _on_end_iteration(self):
        # Print stats in the master process
        if self._verbal and self._state.iteration % self._print_interval == 0:
            self._print_statistics()

    def _on_each_iteration(self):
        with torch.cuda.amp.autocast(enabled=self._use_amp):
            self._state.output = self._state.net(*self._state.inputs)
            self._state.loss = self._criterion(self._state.output, self._state.targets)
        self._state.scaler.scale(self._state.loss).backward()
        self._state.scaler.step(self._state.optimizer)
        self._state.scaler.update()
        self._state.optimizer.zero_grad(set_to_none=True)

    def _print_statistics(self):
        running_loss = self._state.running_loss.mean()
        t_data = self._state.t_data.sum() / self._world_size
        t_iter = self._state.t_iteration.sum() / self._world_size

        # Print stats in the master process
        if self._rank == 0:
            num_iter = len(self._train_loader)
            n_d = len(str(num_iter))
            print(
                "Epoch [{}/{}], Iter. [{}/{}], "
                "Loss: {:.4f}, "
                "Time[Data/Iter.]: [{:.2f}s/{:.2f}s]".format(
                self._state.epoch, self.epochs,
                str(self._state.iteration - num_iter * (self._state.epoch - 1)).zfill(n_d),
                num_iter, running_loss, t_data, t_iter
            ))
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def save_checkpoint(self) -> None:
        """Save a checkpoint of the model state"""
        checkpoint = {
            'iteration': self._state.iteration,
            'epoch': self._state.epoch,
            'model_state_dict': self._state.net.module.state_dict(),
            'optim_state_dict': self._state.optimizer.state_dict(),
            'scaler_state_dict': self._state.scaler.state_dict()
        }
        if self._state.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._state.lr_scheduler.state_dict()
        # Cache the checkpoint
        torch.save(checkpoint, os.path.join(
            self._cache_dir,
            'ckpt_{:05d}_{:02d}.pt'.format(self._state.iteration, self._state.epoch)
        ))
