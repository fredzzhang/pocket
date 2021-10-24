"""
Learning engines under the PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import time
import json
import torch

from torch.nn import Module
from typing import Callable, Iterable, Optional, Any

from ..data import DataDict
from ..ops import relocate_to_device
from ..utils import NumericalMeter, AveragePrecisionMeter

__all__ = [
    'LearningEngine',
    'MultiClassClassificationEngine',
    'MultiLabelClassificationEngine'
]

class State:
    """
    Dict-based state class
    """
    def __init__(self) -> None:
        self._state = DataDict()

    def state_dict(self) -> dict:
        """Return the state dict"""
        return self._state.copy()

    def load_state_dict(self, dict_in: dict) -> None:
        """Load state from external dict"""
        for k in self._state:
            self._state[k] = dict_in[k]

    def fetch_state_key(self, key: str) -> Any:
        """Return a specific key"""
        if key in self._state:
            return self._state[key]
        else:
            raise KeyError("Inexistent key {}".format(key))

    def update_state_key(self, **kwargs) -> None:
        """Override specific keys in the state"""
        for k in kwargs:
            if k in self._state:
                self._state[k] = kwargs[k]
            else:
                raise KeyError("Inexistent key {}".format(k))

class LearningEngine(State):
    r"""
    Base class for learning engine

    By default, all available cuda devices will be used. To disable the usage or
    manually select devices, use the following command:

        CUDA_VISIBLE_DEVICES=, python YOUR_SCRIPT.py
    or
        CUDA_VISIBLE_DEVICES=0,1 python YOUR_SCRIPT.py

    Arguments:

    [REQUIRED ARGS]
        net(Module): The network to be trained
        criterion(callable): Loss function
        train_loader(iterable): Dataloader for training set, with batch input in the
            format [INPUT_1, ..., INPUT_N, LABELS]. Each element should take one of 
            the following forms: Tensor, list[Tensor], dict[Tensor]

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
            net: Module, criterion: Callable, train_loader: Iterable,
            optim: str = 'SGD', optim_params: Optional[dict] = None,
            optim_state_dict: Optional[dict] = None, use_amp: bool = True,
            lr_scheduler: bool = False, lr_sched_params: Optional[dict] = None,
            verbal: bool = True, print_interval: int = 100,
            cache_dir: str = './checkpoints'):

        super().__init__()
        self._dawn = time.time()

        self._device = torch.device('cuda:0') if torch.cuda.is_available() \
            else torch.device('cpu')
        self._multigpu = torch.cuda.device_count() > 1
        self._criterion =  criterion if not isinstance(criterion, torch.nn.Module) \
            else criterion.to(self._device)
        self._train_loader = train_loader
        self._use_amp = use_amp
        self._verbal = verbal
        self._print_interval = print_interval
        self._cache_dir = cache_dir
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        self._state.net = torch.nn.DataParallel(net).to(self._device) if self._multigpu \
            else net.to(self._device)
        # Initialize optimizer
        net_params = [p for p in self._state.net.parameters() if p.requires_grad]
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
                        state[k] = v.to(self._device)
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

        self._state.running_loss = NumericalMeter(maxlen=print_interval)
        # Initialize timers
        self._state.t_data = NumericalMeter(maxlen=print_interval)
        self._state.t_iteration = NumericalMeter(maxlen=print_interval)

    def __call__(self, n: int) -> None:
        self.num_epochs = n
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

                # Force network mode
                self._state.net.train()
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

    def _on_end_epoch(self):
        self.save_checkpoint()
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()

    def _on_start_iteration(self):
        self._state.iteration += 1
        self._state.inputs = relocate_to_device(self._state.inputs, device=self._device)
        self._state.targets = relocate_to_device(self._state.targets, device=self._device)

    def _on_end_iteration(self):
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
        if hasattr(self._train_loader, '__len__'):
            num_iter = len(self._train_loader)
        else:
            num_iter = len(list(self._train_loader))
        n_d = len(str(num_iter))
        print(
            "Epoch [{}/{}], Iter. [{}/{}], "
            "Loss: {:.4f}, "
            "Time[Data/Iter.]: [{:.2f}s/{:.2f}s]".format(
            self._state.epoch, self.num_epochs,
            str(self._state.iteration - num_iter * (self._state.epoch - 1)).zfill(n_d),
            num_iter, self._state.running_loss.mean(),
            self._state.t_data.sum(), self._state.t_iteration.sum())
        )

    def save_checkpoint(self) -> None:
        """Save a checkpoint of the model state"""
        if self._multigpu:
            model_state_dict = self._state.net.module.state_dict()
        else:
            model_state_dict = self._state.net.state_dict()
        checkpoint = {
            'iteration': self._state.iteration,
            'epoch': self._state.epoch,
            'model_state_dict': model_state_dict,
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

    def save_snapshot(self) -> None:
        """Save a snapshot of the engine state"""
        torch.save(self.state_dict(),
            os.path.join(self._cache_dir, 'snapshot_{:05d}_{:02d}.spst'.\
                    format(self._state.iteration, self._state.epoch)))

class MultiClassClassificationEngine(LearningEngine):
    r"""
    Learning engine for multi-class classification problems

    Arguments:

    [REQUIRED ARGS]
        net(Module): The network to be trained
        criterion(Module): Loss function
        train_loader(iterable): Dataloader for training set, with batch input in the
            format [INPUT_1, ..., INPUT_N, LABELS]. Each element should take one of 
            the following forms: Tensor, list[Tensor], dict[Tensor]

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
        >>> from pocket.core import MultiClassClassificationEngine
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
        ...         transform=transforms.Compose([
        ...             transforms.ToTensor(),
        ...             transforms.Normalize((0.1307,), (0.3081,))])
        ...         ),
        ...     batch_size=100, shuffle=False)
        >>> # Intialize learning engine and start training
        >>> engine = MultiClassClassificationEngine(net, criterion, train_loader,
        ...     val_loader=test_loader)
        >>> # Train the network for one epoch with default optimizer option
        >>> # Checkpoints will be saved under ./checkpoints by default, containing 
        >>> # saved model parameters, optimizer statistics and progress
        >>> engine(1)
        """
    def __init__(self,
            net: Module,
            criterion: Callable,
            train_loader: Iterable,
            val_loader: Optional[Iterable] = None,
            **kwargs):

        super().__init__(net, criterion, train_loader, **kwargs)
        if hasattr(val_loader, 'pin_memory'):
            val_loader.pin_memory = torch.cuda.is_available()
        self._val_loader = val_loader
           
    def _validate(self):
        self._state.net.eval()
        correct = 0
        total = 0
        running_loss = NumericalMeter()
        timestamp = time.time()
        for batch in self._val_loader:
            batch = relocate_to_device(batch, device=self._device)
            with torch.no_grad():
                output = self._state.net(*batch[:-1])
            loss = self._criterion(output, batch[-1])
            running_loss.append(loss.item())
            pred = torch.argmax(output, 1)
            correct += torch.eq(pred, batch[-1]).sum().item()
            total += len(pred)
        elapsed = time.time() - timestamp

        print("=> Validation (+{:.2f}s)\n"
            "Epoch: {} | Acc.: {:.4f}[{}/{}] | Loss: {:.4f} | Time: {:.2f}s\n".format(
                time.time() - self._dawn,
                self._state.epoch, correct / total, correct, total,
                running_loss.mean(), elapsed
            ))

    def _on_start_epoch(self):
        if self._state.epoch == 0 and self._val_loader is not None:
            self._validate()
        super()._on_start_epoch()
        self._state.correct = 0
        self._state.total = 0
 
    def _on_end_epoch(self):
        super()._on_end_epoch()
        print("\n=> Training (+{:.2f}s)\n"
            "Epoch: {} | Acc.: {:.4f}[{}/{}]".format(
                time.time() - self._dawn,
                self._state.epoch,
                self._state.correct / self._state.total, self._state.correct, self._state.total
            ))
        if self._val_loader is not None:
            self._validate()

    def _on_end_iteration(self):
        pred = torch.argmax(self._state.output, 1)
        self._state.correct += torch.eq(pred, self._state.targets).sum().item()
        self._state.total += len(pred)
        super()._on_end_iteration()

class MultiLabelClassificationEngine(LearningEngine):
    r"""
    Learning engine for multi-label classification problems

    Arguments:

    [REQUIRED ARGS]
        net(Module): The network to be trained
        criterion(Module): Loss function
        train_loader(iterable): Dataloader for training set, with batch input in the
            format [INPUT_1, ..., INPUT_N, LABELS]. Each element should take one of 
            the following forms: Tensor, list[Tensor], dict[Tensor]

    [OPTIONAL ARGS]
        val_loader(iterable): Dataloader for validation set, with batch input in the
            format [INPUT_1, ..., INPUT_N, LABELS]
        ap_algorithm(str): Choice of algorithm to evaluate average precision. Refer
            to pocket.utils.AveragePrecisionMeter for details
        optim(str): Optimizer to be used. Choose between 'SGD' and 'Adam'
        optim_params(dict): Parameters for the selected optimizer
        optim_state_dict(dict): Optimizer state dict to be loaded
        lr_scheduler(bool): If True, use MultiStepLR as the learning rate scheduler
        lr_sched_params(dict): Parameters for the learning rate scheduler
        verbal(bool): If True, print statistics every fixed interval
        print_interval(int): Number of iterations to print statistics
        cache_dir(str): Directory to save checkpoints

    Example:

        >>> # An example of multi-label classification on voc2012
        >>> CLASSES = (
        ... "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        ... "car", "cat", "chair", "cow", "diningtable", "dog",
        ... "horse", "motorbike", "person", "pottedplant",
        ... "sheep", "sofa", "train", "tvmonitor")
        >>> NUM_CLASSES = len(CLASSES)
        >>> import torch
        >>> from torchvision import datasets, models, transforms
        >>> from pocket.core import MultiLabelClassificationEngine
        >>> # Fix random seed
        >>> torch.manual_seed(0)
        >>> # Initialize network
        >>> net = models.resnet50(num_classes=NUM_CLASSES)
        >>> # Initialize loss function
        >>> criterion = torch.nn.BCEWithLogitsLoss()
        >>> # Prepare dataset
        >>> def target_transform(x):
        ...     target = torch.zeros(NUM_CLASSES)
        ...     anno = x['annotation']['object']
        ...     if isinstance(anno, list):
        ...         for obj in anno:
        ...             target[CLASSES.index(obj['name'])] = 1
        ...     else:
        ...         target[CLASSES.index(anno['name'])] = 1
        ... return target
        >>> train_loader = torch.utils.data.DataLoader(
        ...     datasets.VOCDetection('./data', image_set='train', download=True,
        ...         transform=transforms.Compose([
        ...         transforms.Resize([480, 480]),
        ...         transforms.RandomHorizontalFlip(),
        ...         transforms.ToTensor(),
        ...         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ...         ]),
        ...         target_transform=target_transform),
        ...     batch_size=32, shuffle=True, num_workers=4)
        >>> val_loader = torch.utils.data.DataLoader(
        ...     datasets.VOCDetection('./data', image_set='val',
        ...         transform=transforms.Compose([
        ...         transforms.Resize([480, 480]),
        ...         transforms.ToTensor(),
        ...         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ...         ]),
        ...         target_transform=target_transform),
        ...     batch_size=32, num_workers=4)
        >>> # Initialize learning engine and start training
        >>> engine = MultiLabelClassificationEngine(net, criterion, train_loader,
        ... val_loader=val_loader, print_interval=50,
        ... optim_params={'lr': 0.1, 'momentum': 0.9, 'weight_decay':5e-4})
        >>> # Train the network for one epoch with default optimizer option
        >>> # Checkpoints will be saved under ./checkpoints by default, containing 
        >>> # saved model parameters, optimizer statistics and progress
        >>> engine(1)
        """
    def __init__(self,
            net: Module,
            criterion: Callable,
            train_loader: Iterable,
            val_loader: Optional[Iterable] = None,
            ap_algorithm: str = 'INT',
            **kwargs):
        
        super().__init__(net, criterion, train_loader, **kwargs)
        if hasattr(val_loader, 'pin_memory'):
            val_loader.pin_memory = torch.cuda.is_available()
        self._val_loader = val_loader
        self._ap_alg = ap_algorithm

        self.ap = dict()

    def _validate(self):
        self._state.net.eval()
        meter = AveragePrecisionMeter(algorithm=self._ap_alg)
        running_loss = NumericalMeter()
        timestamp = time.time()
        for batch in self._val_loader:
            batch = relocate_to_device(batch, device=self._device)
            with torch.no_grad():
                output = self._state.net(*batch[:-1])
            loss = self._criterion(output, batch[-1])
            running_loss.append(loss.item())
            meter.append(output, batch[-1])
        ap = meter.eval()
        elapsed = time.time() - timestamp

        print("=> Validation (+{:.2f}s)\n"
            "Epoch: {} | mAP: {:.4f} | Loss: {:.4f} | Time: {:.2f}s\n".format(
                time.time() - self._dawn,
                self._state.epoch, ap.mean().item(),
                running_loss.mean(), elapsed
            ))
        self.ap[self._state.epoch] = ap.tolist()
        with open(os.path.join(self._cache_dir, 'ap.json'), 'w') as f:
            json.dump(self.ap, f)

    def _on_start_epoch(self):
        if self._state.epoch == 0 and self._val_loader is not None:
            self._validate()
            self._state.ap = AveragePrecisionMeter(algorithm=self._ap_alg)
        super()._on_start_epoch()

    def _on_end_epoch(self):
        super()._on_end_epoch()
        timestamp = time.time()
        ap = self._state.ap.eval()
        elapsed = time.time() - timestamp
        print("\n=> Training (+{:.2f}s)\n"
            "Epoch: {} | mAP: {:.4f} | Time(eval): {:.2f}s".format(
                time.time() - self._dawn,
                self._state.epoch,
                ap.mean().item(), elapsed
            ))

        self._state.ap.reset()
        if self._val_loader is not None:
            self._validate()

    def _on_end_iteration(self):
        self._state.ap.append(self._state.output, self._state.targets)
        super()._on_end_iteration()
