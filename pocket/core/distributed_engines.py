"""
Distributed learning engines based on torch.distributed

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import time
import torch
import torch.utils.data as Tdata
import torch.distributed as dist
import torch.multiprocessing as mp

from ..data import DataDict
from ..ops import relocate_to_cuda

class DistributedLearningEngine:
    r"""
    """
    def __init__(self, 
            net, criterion, train_loader,
            # Learning related parameters
            optim='SGD', optim_params=None, optim_state_dict=None,
            lr_scheduler=False, lr_sched_params=None,
            verbal=True, print_interval=100, cache_dir='./checkpoints',
            # Multiprocessing related parameters
            backend='nccl', init_method='env://',
            master_addr='127.0.0.1', master_port='29500'):
        
        self._dawn = time.time()
        assert torch.cuda.is_available(), \
            "Torch not compiled with CUDA enabled"
        if torch.cuda.device_count() == 1:
            print("WARNING: Number of available CUDA devices is 1. "
                "Distributed training is not necessary")
        torch.backends.cudnn.benchmark = True

        self._devices = [idx for idx in range(torch.cuda.device_count())]
        self._world_size = torch.cuda.device_count()
        # Replicate network to every CUDA device
        replicas = torch.nn.parallel.repliace(net, self._devices)
        self._replicas = [torch.nn.parallel.DistributedDataParallel(
            replicas[idx], device_ids=[idx]) for idx in self._devices]

        self._criterion = criterion
        self._train_loader = Tdata.DataLoader(
            train_loader.dataset,
            batch_size=int(train_loader.batch_size / self._world_size),
            num_workers=train_loader.num_workers,
            pin_memory=True,
            drop_last=train_loader.drop_last,
            sampler=Tdata.distributed.DistributedSampler(train_loader.dataset)
        )
        self._verbal = verbal
        self._print_interval = print_interval
        self._cache_dir = cache_dir

        self._backend = backend
        self._init_method = init_method
        self._master_addr = master_addr
        self._master_port = master_port

        # Prepare optimizer parameters
        if optim_params is None:
            optim_params = {
                    'lr': 0.001,
                    'momentum': 0.9,
                    'weight_decay': 5e-4
            } if optim == 'SGD' else {'lr': 0.001, 'weight_decay': 5e-4}
        net_params = [p for p in self._replicas[0].parameters() if p.requires_grad]
        self._optimizer = torch.optim.SGD(net_params, **optim_params)\
            if optim == 'SGD' \
            else torch.optim.Adam(net_params, **optim_params)
        # Load optimzer state dict if provided
        if optim_state_dict is not None:
            self._optimizer.load_state_dict(optim_state_dict)
            # Relocate optimizer state to designated device
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self._devices[0])
        # Initialize optimizers for every model replica
        # self._optimizers = []
        # for idx, replica in enumerate(self._replicas):
        #     params_with_grad = [p for p in replica.parameters() if p.requires_grad]
        #     each_optim = optim_constructor(params_with_grad, **optim_params)
        #     if optim_state_dict is not None:
        #         each_optim.load_state_dict(optim_state_dict)
        #         # Relocate optimizer state to designated deivce
        #         for state in each_optim.state.values():
        #             for k, v in state.items():
        #                 if isinstance(v, torch.Tensor):
        #                     state[k] = v.cuda(idx)

        # Initialize learning rate scheduler
        lr_sched_params = {
                'milestones': [50,100],
                'gamma': 0.1
            } if lr_sched_params is None else lr_sched_params
        self._lr_scheduler = None if not lr_scheduler \
            else torch.optim.lr_scheduler.MultiStepLR(self._optimizer, 
                **lr_sched_params)
        # Initialize logger

        self.epoch = 0
        self.iteration = 0

    def __call__(self, n):
        # Spawn subprocesses and train for a specified number of epochs
        mp.spawn(self._main,
            args=(n, self.epoch, self.iteration, 
            self._init_process_group,
            self._replicas, self._criterion, self._train_loader,
            self._optimizer, self._lr_scheduler,
            self._verbal, self._print_interval, self._cache_dir), 
            nprocs=self._world_size)
        self.epoch += n

    def _init_process_group(self, rank):
        # Set environment variables
        os.environ['MASTER_ADDR'] = self._master_addr
        os.environ['MASTER_PORT'] = self._master_port

        # Initialize the process group
        dist.init_process_group(self._backend, 
            init_method=self._init_method, 
            rank=rank, 
            world_size=self._world_size)

    @staticmethod
    def _cleanup():
        dist.destroy_process_group()

    @classmethod
    def _main(cls, rank, n, epoch, iteration,
            init_process_group,
            replicas, criterion, train_loader,
            optimizer, lr_scheduler,
            verbal, print_interval, cache_dir):
        # Combine all state variables
        state = DataDict({
            'rank': rank,
            'epoch': epoch,
            'iteration': iteration,
            'net': replicas[rank],
            'criterion': criterion,
            'train_loader': train_loader,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'verbal': verbal,
            'print_interval': print_interval,
            'cache_dir': cache_dir
        })
        init_process_group(rank)
        for _ in range(n):
            cls._on_start_epoch(state)
            for batch in train_loader:
                state.input = batch[:-1]
                state.target = batch[-1]
                cls._on_start_iteration(state)
                cls._on_each_iteration(state)
                cls._on_end_iteration(state)
            cls._on_end_epoch(state)
        cls._cleanup()

    @staticmethod
    def _on_start_epoch(ctx):
        ctx.epoch += 1

    @staticmethod
    def _on_end_epoch(ctx):
        if ctx.lr_scheduler is not None and ctx.rank == 0:
            ctx.lr_scheduler.step()

    @staticmethod
    def _on_start_iteration(ctx):
        ctx.iteration += 1
        ctx.input = relocate_to_cuda(ctx.input, ctx.rank)
        ctx.target = relocate_to_cuda(ctx.target, ctx.rank)

    @staticmethod
    def _on_end_iteration(ctx):
        pass

    @staticmethod
    def _on_each_iteration(ctx):
        if ctx.rank == 0:
            ctx.optimizer.zero_grad()
        ctx.output = ctx.net(*ctx.input)
        ctx.loss = ctx.criterion(ctx.output, ctx.target)
        ctx.loss.backward()
        if ctx.rank == 0:
            ctx.optimizer.step()
