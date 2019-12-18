"""
Learning engines under the PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import copy
import time
import torch
from torch.utils.data import DataLoader

from ..data import DataDict
from ..utils import NumericalMeter, HandyTimer, AveragePrecisionMeter

__all__ = [
    'LearningEngine',
    'MultiClassClassificationEngine',
    'MultiLabelClassificationEngine'
]

class State:
    """
    Dict-based state class
    """
    def __init__(self):
        self._state = DataDict()

    def state_dict(self):
        """Return the state dict"""
        return self._state.copy()

    def load_state_dict(self, dict_in):
        """Load state from external dict"""
        for k in self._state:
            self._state[k] = dict_in[k]

    def fetch_state_key(self, key):
        """Return a specific key"""
        if key in self._state:
            return self._state[key]
        else:
            raise KeyError("Inexistent key {}".format(key))

    def update_state_key(self, **kwargs):
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
            net, criterion, train_loader,
            optim='SGD', optim_params=None, optim_state_dict=None,
            lr_scheduler=False, lr_sched_params=None,
            verbal=True, print_interval=100, cache_dir='./checkpoints'):

        super().__init__()
        self._dawn = time.time()

        self._device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')
        self._multigpu = torch.cuda.device_count() > 1
        self._criterion =  criterion if not isinstance(criterion, torch.nn.Module) \
            else criterion.to(self._device)
        self._train_loader = train_loader
        self._verbal = verbal
        self._print_interval = print_interval
        self._cache_dir = cache_dir

        # Set flags for GPU
        torch.backends.cudnn.benchmark = torch.cuda.is_available()
        if hasattr(train_loader, 'pin_memory'):
            train_loader.pin_memory = torch.cuda.is_available()

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
        self._state.optimizer = torch.optim.SGD(net_params, **optim_params)\
            if optim == 'SGD' \
            else torch.optim.Adam(net_params, **optim_params)
        # Load optimzer state dict if provided
        if optim_state_dict is not None:
            self._state.optimizer.load_state_dict(optim_state_dict)
            # Relocate optimizer state to designated device
            for state in self._state.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self._device)
        self._state.epoch = 0
        self._state.iteration = 0

        # Initialize learning rate scheduler
        lr_sched_params = {
                'milestones': [50,100],
                'gamma': 0.1
            } if lr_sched_params is None else lr_sched_params
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

    def _relocate_to_device(self, x):
        if type(x) is torch.Tensor:
            return x.to(self._device)
        elif type(x) is list:
            # Assume input data is a list of tensors
            return [item.to(self._device) for item in x]
        elif type(x) is dict:
            # Assume input data is a dictionary of tensors
            for key in x:
                x[key] = x[key].to(self._device)
            return x
        else:
            raise TypeError('Unsupported type of data {}'.format(type(x)))

    def _on_start_epoch(self):
        self._state.epoch += 1

    def _on_end_epoch(self):
        self.save_checkpoint()
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    def _on_start_iteration(self):
        self._state.iteration += 1
        self._state.input = [self._relocate_to_device(item) for item in self._state.input]
        self._state.target = self._relocate_to_device(self._state.target)

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
        print("[Ep.][Iter.]: [{}][{}] | "
                "Loss: {:.4f} | "
                "Time[Data][Iter.]: [{:.4f}s][{:.4f}s]".format(
                    self._state.epoch, self._state.iteration,
                    self._state.running_loss.mean(),
                    self._state.t_iteration.sum(),
                    self._state.t_data.sum())
            )
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def save_checkpoint(self):
        """Save a checkpoint of the model state"""
        if not os.path.exists(self._cache_dir):
            os.mkdir(self._cache_dir)
        # Make a copy of the network parameters and relocate to cpu
        model_state_dict = \
            self._state.net.module.state_dict().copy() if self._multigpu \
            else self._state.net.state_dict().copy()
        for k in model_state_dict:
            model_state_dict[k] = model_state_dict[k].cpu()
        # Make a copy of the optimizer and relocate to cpu
        optim_copy = copy.deepcopy(self._state.optimizer)
        for state in optim_copy.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self._device)
        torch.save({
            'iteration': self._state.iteration,
            'epoch': self._state.epoch,
            'model_state_dict': model_state_dict,
            'optim_state_dict': optim_copy.state_dict()
            }, os.path.join(self._cache_dir, 'ckpt_{:05d}_{:02d}.pt'.\
                    format(self._state.iteration, self._state.epoch)))

    def save_snapshot(self):
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

        => Validation (+5.13s)
        Epoch: 0 | Acc.: 0.1008[1008/10000] | Loss: 2.3036 | Time: 2.35s

        [Ep.][Iter.]: [1][100] | Loss: 2.2971 | Time[Data][Iter.]: [2.9884s][2.8512s]
        [Ep.][Iter.]: [1][200] | Loss: 2.2773 | Time[Data][Iter.]: [0.2582s][2.8057s]
        [Ep.][Iter.]: [1][300] | Loss: 2.2289 | Time[Data][Iter.]: [0.2949s][2.9972s]
        [Ep.][Iter.]: [1][400] | Loss: 2.0143 | Time[Data][Iter.]: [0.2578s][2.4794s]

        => Training (+17.66s)
        Epoch: 1 | Acc.: 0.3181[19090/60000]
        => Validation (+19.43s)
        Epoch: 1 | Acc.: 0.7950[7950/10000] | Loss: 0.7701 | Time: 2.04s
    """
    def __init__(self,
            net,
            criterion,
            train_loader,
            val_loader=None,
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
            batch = [self._relocate_to_device(item) for item in batch]
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
        super()._on_end_iteration()
        pred = torch.argmax(self._state.output, 1)
        self._state.correct += torch.eq(pred, self._state.target).sum().item()
        self._state.total += len(pred)

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

        => Validation (+57.05s)
        Epoch: 0 | mAP: 0.0888 | Loss: 6.4674 | Time: 54.01s

        [Ep.][Iter.]: [1][50] | Loss: 0.3601 | Time[Data][Iter.]: [26.5870s][0.9164s]
        [Ep.][Iter.]: [1][100] | Loss: 0.2634 | Time[Data][Iter.]: [19.1039s][0.0111s]
        [Ep.][Iter.]: [1][150] | Loss: 0.2532 | Time[Data][Iter.]: [19.2337s][0.0104s]

        => Training (+195.74s)
        Epoch: 1 | mAP: 0.0925 | Time(eval): 2.35s
        => Validation (+238.64s)
        Epoch: 1 | mAP: 0.1283 | Loss: 0.4617 | Time: 42.90s
    """
    def __init__(self,
            net,
            criterion,
            train_loader,
            val_loader=None,
            ap_algorithm='INT',
            **kwargs):
        
        super().__init__(net, criterion, train_loader, **kwargs)
        if hasattr(val_loader, 'pin_memory'):
            val_loader.pin_memory = torch.cuda.is_available()
        self._val_loader = val_loader
        self._ap_alg = ap_algorithm

    def _validate(self):
        self._state.net.eval()
        ap = AveragePrecisionMeter(algorithm=self._ap_alg)
        running_loss = NumericalMeter()
        timestamp = time.time()
        for batch in self._val_loader:
            batch = [self._relocate_to_device(item) for item in batch]
            with torch.no_grad():
                output = self._state.net(*batch[:-1])
            loss = self._criterion(output, batch[-1])
            running_loss.append(loss.item())
            ap.append(output, batch[-1])
        map_ = ap.eval().mean().item()
        elapsed = time.time() - timestamp

        print("=> Validation (+{:.2f}s)\n"
            "Epoch: {} | mAP: {:.4f} | Loss: {:.4f} | Time: {:.2f}s\n".format(
                time.time() - self._dawn,
                self._state.epoch, map_,
                running_loss.mean(), elapsed
            ))

    def _on_start_epoch(self):
        if self._state.epoch == 0 and self._val_loader is not None:
            self._validate()
            self._state.ap = AveragePrecisionMeter(algorithm=self._ap_alg)
        super()._on_start_epoch()

    def _on_end_epoch(self):
        super()._on_end_epoch()
        timestamp = time.time()
        map_ = self._state.ap.eval().mean().item()
        elapsed = time.time() - timestamp
        print("\n=> Training (+{:.2f}s)\n"
            "Epoch: {} | mAP: {:.4f} | Time(eval): {:.2f}s".format(
                time.time() - self._dawn,
                self._state.epoch,
                map_, elapsed
            ))
        self._state.ap.reset()
        if self._val_loader is not None:
            self._validate()

    def _on_end_iteration(self):
        super()._on_end_iteration()
        self._state.ap.append(self._state.output, self._state.target)
