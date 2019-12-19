"""
Distributed learning engines based on torch.distributed

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import time
import copy
import torch
import torch.utils.data as Tdata
import torch.distributed as dist
import torch.multiprocessing as mp

from ..data import DataDict
from ..ops import relocate_to_cuda
from ..utils import SyncedNumericalMeter

class DistributedLearningEngine:
    r"""
    Distributed learning engine based on torch.distributed

    Arguments:

    [REQUIRED ARGS]
        net(Module): The network to be trained
        criterion(callable): Loss function
        train_loader(iterable): Dataloader for training set, with batch input in the
            format [INPUT_1, ..., INPUT_N, LABELS]. Each element should take one of 
            the following forms: Tensor, list[Tensor], dict[Tensor]. 
            
        NOTE: The dataloader passed into the engine is merely used as a container of
        parameters. The actual batch size used for distributed dataloader will be 
        divided by the number of subprocesses. Sampler or batch sampler will
        be ignored automatically.

    [OPTIONAL ARGS]
        optim(str): Optimizer to be used. Choose between 'SGD' and 'Adam'
        optim_params(dict): Parameters for the selected optimizer
        optim_state_dict(dict): Optimizer state dict to be loaded
        lr_scheduler(bool): If True, use MultiStepLR as the learning rate scheduler
        lr_sched_params(dict): Parameters for the learning rate scheduler
        verbal(bool): If True, print statistics every fixed interval
        print_interval(int): Number of iterations to print statistics
        cache_dir(str): Directory to save checkpoints
        backend(str): A choice between 'nccl', 'gloo' and 'mpi'
        init_method(str): URL specifying how to initialize the process group
        master_addr(str): IP address for master process
        master_port(str): Communication port for master process
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
        # Initialize optimizer for the model replica to be assigned to master process
        # Other model replicas will be in sync during forward pass
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

        # Initialize learning rate scheduler
        lr_sched_params = {
                'milestones': [50,100],
                'gamma': 0.1
            } if lr_sched_params is None else lr_sched_params
        self._lr_scheduler = None if not lr_scheduler \
            else torch.optim.lr_scheduler.MultiStepLR(self._optimizer, 
                **lr_sched_params)

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
            initialize,
            replicas, criterion, train_loader,
            optimizer, lr_scheduler,
            verbal, print_interval, cache_dir):
        # Combine all state variables
        state = DataDict({
            'rank': rank, 'epoch': epoch, 'iteration': iteration,
            'net': replicas[rank], 'criterion': criterion, 'train_loader': train_loader,
            'optimizer': optimizer, 'lr_scheduler': lr_scheduler,
            'verbal': verbal, 'print_interval': print_interval, 'cache_dir': cache_dir
        })
        # Initialize process group
        initialize(rank)

        state.running_loss = SyncedNumericalMeter(maxlen=print_interval)
        # Initialize timers
        state.t_data = SyncedNumericalMeter(maxlen=print_interval)
        state.t_iteration = SyncedNumericalMeter(maxlen=print_interval)
        # Training loop
        for _ in range(n):
            cls._on_start_epoch(state)
            timestamp = time.time()
            for batch in train_loader:
                state.input = batch[:-1]
                state.target = batch[-1]
                state.t_data.append(time.time() - timestamp)

                cls._on_start_iteration(state)
                # Force network mode
                state.net.train()
                cls._on_each_iteration(state)
                state.running_loss.append(state.loss.item())
                cls._on_end_iteration(state)
                state.t_iteration.append(time.time() - timestamp)
                timestamp = time.time()

            cls._on_end_epoch(state)
        # Garbage collecting
        cls._cleanup()

    @staticmethod
    def _on_start_epoch(ctx):
        ctx.epoch += 1

    @classmethod
    def _on_end_epoch(cls, ctx):
        if ctx.rank == 0:
            # Save the model replica in master process
            cls._save_checkpoint(ctx)
            if ctx.lr_scheduler is not None:
                ctx.lr_scheduler.step()

    @staticmethod
    def _on_start_iteration(ctx):
        ctx.iteration += 1
        ctx.input = relocate_to_cuda(ctx.input, ctx.rank)
        ctx.target = relocate_to_cuda(ctx.target, ctx.rank)

    @classmethod
    def _on_end_iteration(cls, ctx):
        if ctx.verbal and ctx.iteration % ctx.print_interval == 0:
            cls._print_statistics(ctx)

    @staticmethod
    def _on_each_iteration(ctx):
        if ctx.rank == 0:
            ctx.optimizer.zero_grad()
        ctx.output = ctx.net(*ctx.input)
        ctx.loss = ctx.criterion(ctx.output, ctx.target)
        ctx.loss.backward()
        if ctx.rank == 0:
            ctx.optimizer.step()

    @staticmethod
    def _print_statistics(ctx):
        loss = ctx.running_loss.mean()
        time_iter = ctx.t_iteration.sum()
        time_data = ctx.t_data.sum()
        if ctx.rank == 0:
            print("[Ep.][Iter.]: [{}][{}] | "
                "Loss: {:.4f} | "
                "Time[Data][Iter.]: [{:.4f}s][{:.4f}s]".format(
                    ctx.epoch, ctx.iteration,
                    loss, time_data, time_iter)
            )
        ctx.t_iteration.reset()
        ctx.t_data.reset()
        ctx.running_loss.reset()

    @staticmethod
    def _save_checkpoint(ctx):
        """Save a checkpoint of the model state"""
        if not os.path.exists(ctx.cache_dir):
            os.mkdir(ctx.cache_dir)
        # Make a copy of the network parameters and relocate to cpu
        model_state_dict = ctx.net.module.state_dict().copy()
        for k in model_state_dict:
            model_state_dict[k] = model_state_dict[k].cpu()
        # Make a copy of the optimizer and relocate to cpu
        optim_copy = copy.deepcopy(ctx.optimizer)
        for state in optim_copy.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        torch.save({
            'iteration': ctx.iteration,
            'epoch': ctx.epoch,
            'model_state_dict': model_state_dict,
            'optim_state_dict': optim_copy.state_dict()
            }, os.path.join(ctx.cache_dir, 'ckpt_{:05d}_{:02d}.pt'.\
                    format(ctx.iteration, ctx.epoch)))