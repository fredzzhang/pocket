"""
Engines for training and testing under the PyTorch framework

Written by Frederic Zhang
Australian National University

Last updated in Oct. 2019
"""

import os
import copy
import torch
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .io import Log, TrainLog, load_pkl_from
from ..data import DataDict, ParallelOnlineBatchSampler, IndexSequentialSampler

class Engine:
    """
    Base class
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
            self._state[key] = val
        else:
            raise KeyError('Inexistent key {}'.format(key))

class BubbleTrainer(Engine):
    """
    Learning engine for bubble training

    Arguments:

    [REQUIRED ARGS]
        net(Module): The network to be trained
        loss_fn(Module): Loss function
        dataset(indexable): Dataset for training, with batch output in the format
            [INPUT_1, ..., INPUT_N, LABELS]. Otherwise, use {input_transform}.
        batch_size(int): Batch size

    [OPTIONAL ARGS]
        update_rule(str): 
        num_workers(int): Number of workers used to load data
        cache_dir(str): Diretory to saved models and training log, default: './'
        print_interval(int): Number of steps to print training and validation losses, default: 200
        optim(str): Optimizer used in training, 'SGD', 'Adam'
        optim_params(dict): Optimizer parameter dict with same keywords as selected optimizer
            default: {'lr':0.1, 'momentum':0.9, 'weight_decay':5e-4}
        device(str): Primary device, e.g. 'cpu', 'cuda:0'
        multi_gpu(bool): Allows multi-GPU computation. When being true, {device} has to be 'cuda:0'
        input_transform(handle): Transform the batch data to desired format
            default: lambda a: a -> keep the original format 
    """
    def __init__(self,
            net,
            loss_fn,
            dataset,
            batch_size,
            update_rule='multiclass',
            shuffle=False,
            num_workers=4,
            cache_dir=None,
            optim='SGD',
            optim_params={
                'lr': 0.001,
                'momentum': 0.9,
                'weight_decay': 5e-4},
            device='cpu',
            multi_gpu=False,
            input_transform=lambda a: a
            ):

        super(BubbleTrainer, self).__init__()

        # declare primary device
        self._state.device = torch.device(device)
        # relocate network
        if multi_gpu:
            assert device == 'cuda:0',\
                    'The primary device has to be \'cuda:0\' to enable multiple gpus'
            net = torch.nn.DataParallel(net)
        self._state.net = net.to(self._state.device)
        # relocate loss function
        self._state.criterion = loss_fn.to(self._state.device)
        # initialize optimizer
        if optim == 'SGD':
            self._state.optimizer = torch.optim.SGD(self._state.net.parameters(),
                    **optim_params)
        elif optim == 'Adam':
            self._state.optimizer = torch.optim.Adam(self._state.net.paramters(),
                    **optim.params)
        else:
            raise ValueError('Unsupported optimizer type {}'.format(optim))
        # initilize logger
        if cache_dir is None:
            self._state.logger = None
        else:
            self._state.logger = TrainLog(os.path.join(cache_dir, 'train_log'))
        
        self._state.step = 0
        self._state.epoch = 0
        self._state.batch_size = batch_size
        self._state.shuffle = shuffle
        self._state.num_workers = num_workers
        self._state.multi_gpu = multi_gpu
        self._state.cache_dir = cache_dir
        self._state.dataset = dataset
        self._state.input_transform = input_transform

        if update_rule == 'multiclass':
            self._update_method = self._update_multiclass
        elif update_rule == 'multilabel':
            self._update_method = self._update_multilabel_gnorm
        else:
            raise ValueError('Unsupported update rule {}'.format(update_rule))

        self._state.subset = np.random.permutation(len(dataset))

    def _update_multilabel_L2_sort(self, logits, labels):
        """
        Arguments:
            logits(CUDA tensor[N, C])
            labels(tensor[N, C])
        """
        order = torch.argsort(logits, 0).cpu()
        # find the subset of unsorted examples
        subset = []
        for cls in range(logits.shape[1]):
            l_idx = torch.min(torch.nonzero(labels[order[:, cls], cls]))
            h_idx = torch.max(torch.nonzero(1 - labels[order[:, cls], cls]))
            subset.append(order[l_idx:h_idx + 1, cls])
        subset = torch.cat(subset, 0).unique()

        return subset[torch.sum(logits[subset, :] ** 2, 1).argsort().cpu()].numpy()


    def _update_multilabel_loss_sort(self, logits, labels):
        """
        Arguments:
            logits(CUDA tensor[N, C])
            labels(tensor[N, C])
        """
        order = torch.argsort(logits, 0).cpu()
        # find the subset of unsorted examples
        subset = []
        for cls in range(logits.shape[1]):
            l_idx = torch.min(torch.nonzero(labels[order[:, cls], cls]))
            h_idx = torch.max(torch.nonzero(1 - labels[order[:, cls], cls]))
            subset.append(order[l_idx:h_idx + 1, cls])
        subset = torch.cat(subset, 0).unique()
        labels = (labels * 2 - 1).to(logits.device)

        return subset[torch.sum(logits[subset, :] * labels[subset, :], 1).argsort().cpu()].numpy()

    def _update_multilabel_simple_merge(self, logits, labels):
        """
        Arguments:
            logits(CUDA tensor[N, C])
            labels(tensor[N, C])
        """
        order = torch.argsort(logits, 0).cpu()
        # find the subset of unsorted examples
        subset = []
        for cls in range(logits.shape[1]):
            l_idx = torch.min(torch.nonzero(labels[order[:, cls], cls]))
            h_idx = torch.max(torch.nonzero(1 - labels[order[:, cls], cls]))
            subset.append(order[l_idx:h_idx + 1, cls])
        return torch.cat(subset, 0).unique().numpy()

    def _update_multilabel_merge_rand(self, logits, labels):
        subset = self._update_multilabel_simple_merge(logits, labels)
        return subset[np.random.permutation(len(subset))]

    def _update_multilabel_gnorm(self, logits, labels):
        gnorm = torch.abs(logits - labels.to(logits.device)).sum(1)
        order = torch.argsort(gnorm).cpu().numpy()
        return order[round(len(order) * 0.2):]

    def _update_multiclass_a(self, logits, labels):
        """
        Arguments:
            logits(CUDA tensor[N, C])
            labels(tensor[N])
        """
        b_labels = torch.zeros(logits.shape)
        b_labels[torch.linspace(0, len(labels) - 1, len(labels)).long(), labels.long()] = 1
        return self._update_multilabel_merge_rand(logits, b_labels)

    def _update_multiclass(self, logits, labels):
        pred = torch.argmax(logits, 1).cpu()
        subset = np.where((pred != labels).numpy())[0]
        return subset[np.random.permutation(len(subset))]

    def reset(self):
        self._state.subset = np.linspace(
                0,
                len(self._state.dataset) - 1,
                len(self._state.dataset)).astype(np.int32)

    def __call__(self, nepoch):

        for _ in range(nepoch):
            acc_loss = 0

            # use one-based index for epoch number
            self._state.epoch += 1

            if self._state.logger is not None:
                self._state.logger.time()
                self._state.logger.write('\nEPOCH: {}\n'.format(self._state.epoch))
                self._state.logger.write('Learning rate: {}, step: {}\n\n'.format(
                    self._state.optimizer.state_dict()['param_groups'][0]['lr'],
                    self._state.step))

            self._state.net.train()
#            if self._state.multi_gpu:
#                self._state.net.module.activate_bn()
#            else:
#                self._state.net.activate_bn()

#            if self._state.epoch % 2:
            if 1:
                # generate a random permutation array
#                idx_perm = np.random.permutation(len(self._state.dataset))
                sampler = IndexSequentialSampler(self._state.subset)

                dataloader = torch.utils.data.DataLoader(
                        self._state.dataset,
                        batch_size=self._state.batch_size,
                        num_workers=self._state.num_workers,
                        sampler=sampler,
                        pin_memory=self._state.device.type=='cuda'
                        )
                logits = []
                labels = []
                for dtuple in dataloader:
                    # apply transform to input data
                    dtuple = self._state.input_transform(dtuple)
                    # relocate tensors to designated device
                    dtuple = [item.to(self._state.device) for item in dtuple]
                    # forward pass
                    out = self._state.net(*dtuple[:-1])
                    # compute loss
                    loss = self._state.criterion(out, dtuple[-1])
                    self._state.optimizer.zero_grad()
                    # back propogation
                    loss.backward()
                    # update weights
                    self._state.optimizer.step()

                    self._state.step += 1

                    acc_loss += loss.item()
                    logits.append(out.detach())
                    labels.append(dtuple[-1].cpu())

                logits = torch.softmax(torch.cat(logits, 0), 1)\
                        if self._update_method == self._update_multiclass\
                        else torch.sigmoid(torch.cat(logits, 0))
                labels = torch.cat(labels, 0)
                # update subset
                self._state.subset = self._state.subset[self._update_method(logits, labels)]

###
#                logits_n = []
#                labels_n = []
#                self._state.net.eval()
#                for dtuple in dataloader:
#                    dtuple = self._state.input_transform(dtuple)
#                    dtuple = [item.to(self._state.device) for item in dtuple]
#                    with torch.no_grad():
#                        out = self._state.net(*dtuple[:-1])
#                    logits_n.append(out.cpu())
#                    labels_n.append(dtuple[-1].cpu())
#                logits_n = torch.cat(logits_n, 0)
#                labels_n = torch.cat(labels_n, 0)
#                subset_n = idx_perm[self._update_method(logits_n, labels_n)]
#
#                self._state.mine = [
#                        len(self._state.subset),
#                        len(subset_n),
#                        np.sum(np.isin(subset_n, self._state.subset))
#                        ]
###

                # log accumulated loss
                if self._state.logger is not None:
                    self._state.logger.log(self._state.step, acc_loss)
            else:
                # set BN to eval mode and use tracked mean and var.
#                if self._state.multi_gpu:
#                    self._state.net.module.freeze_bn()
#                else:
#                    self._state.net.freeze_bn()

                idx_perm = np.random.permutation(len(self._state.dataset))
                sampler = IndexSequentialSampler(idx_perm)

                dataloader = torch.utils.data.DataLoader(
                        self._state.dataset,
                        batch_size=self._state.batch_size,
                        num_workers=self._state.num_workers,
                        sampler=sampler,
                        pin_memory=self._state.device.type=='cuda'
                        )

                logits = []
                labels = []
                self._state.net.eval()
                for dtuple in dataloader:
                    dtuple = self._state.input_transform(dtuple)
                    dtuple = [item.to(self._state.device) for item in dtuple]
                    with torch.no_grad():
                        out = self._state.net(*dtuple[:-1])
                    logits.append(out.cpu())
                    labels.append(dtuple[-1].cpu())
                logits = torch.cat(logits, 0)
                labels = torch.cat(labels, 0)
                subset = idx_perm[self._update_method(logits, labels)]

                self._state.kept = np.isin(subset, self._state.subset)
                self._state.subset = subset
               

                sampler = torch.utils.data.SubsetRandomSampler(self._state.subset) \
                        if self._state.shuffle else IndexSequentialSampler(self._state.subset) 
                dataloader = torch.utils.data.DataLoader(
                        self._state.dataset,
                        batch_size=self._state.batch_size,
                        num_workers=self._state.num_workers,
                        sampler=sampler,
                        pin_memory=self._state.device.type=='cuda'
                        )
                self._state.net.train()
                for dtuple in dataloader:
                    dtuple = self._state.input_transform(dtuple)
                    dtuple = [item.to(self._state.device) for item in dtuple]
                    out = self._state.net(*dtuple[:-1])
                    loss = self._state.criterion(out, dtuple[-1])
                    self._state.optimizer.zero_grad()
                    loss.backward()
                    self._state.optimizer.step()

                    self._state.step += 1


            # save checkpoint at the end of each epoch
            if self._state.cache_dir is not None:
                if self._state.multi_gpu:
                    net_copy = copy.deepcopy(self._state.net.module).cpu()
                else:
                    net_copy = copy.deepcopy(self._state.net).cpu()
                # copy the optimizer
                optim_copy = copy.deepcopy(self._state.optimizer)
                for state in optim_copy.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()
                torch.save({
                    'step': self._state.step,
                    'epoch': self._state.epoch,
                    'model_state_dict': net_copy.state_dict(),
                    'optim_state_dict': optim_copy.state_dict()
                    }, os.path.join(self._state.cache_dir, 'torch_ckpt_s{:05d}_e{:02d}'.\
                            format(self._state.step, self._state.epoch)))


class NetTrainer(Engine):
    r"""
    Network trainer class

    Arguments:
    
    [REQUIRED ARGS]
        net(Module): The network to be trained
        loss_fn(Module): Loss function
        train_loader(iterable): Dataloader for training set, with batch output in the
            format [INPUT_1, ..., INPUT_N, LABELS]. Otherwise, use {input_transform}

    [OPTIONAL ARGS]
        cache_dir(str): Diretory to saved models and training log, default: './'
        print_interval(int): Number of steps to print training and validation losses, default: 200
        optim(str): Optimizer used in training, 'SGD', 'Adam'
        optim_params(dict): Optimizer parameter dict with same keywords as selected optimizer
            default: {'lr':0.1, 'momentum':0.9, 'weight_decay':5e-4}
        optim_state_dict(OrderedDict): State dict of the optimizer
        device(str): Primary device, e.g. 'cpu', 'cuda:0'
        multi_gpu(bool): Allows multi-GPU computation. When being true, {device} has to be 'cuda:0'
        val_loader(iterable): Dataloader for validation set with the same data format as train loader
        lr_scheduler(bool): Use learning rate scheduler, default: True
        sched_params(dict): Learning rate scheduler parameter dict
        sched_ext_update(bool): If True, disable updating the scheduler with running loss, assuming
            update will be done externally
        input_transform(handle): Transform the batch data to desired format
            default: lambda a: a -> keep the original format 

    Example:

        >>> import torch
        >>> from pocket.utils import NetTrainer
        >>> net = torch.nn.Linear(1, 1)
        >>> loss_fn = torch.nn.BCEWithLogitsLoss()
        >>> dataloader = [[torch.tensor([0.8]), torch.tensor([1.])],
        ... [torch.tensor([0.2]), torch.tensor([0.])]]
        >>> trainer = NetTrainer(net, loss_fn, dataloader)
        >>> trainer(10)
    """
    def __init__(self,
            net,
            loss_fn,
            train_loader,
            cache_dir=None,
            print_interval=200,
            optim='SGD',
            optim_params={
                'lr': 0.001,
                'momentum': 0.9,
                'weight_decay': 5e-4},
            optim_state_dict=None,
            device='cpu',
            multi_gpu=False,
            val_loader=None,
            loss_fn_val=None,
            lr_scheduler=False,
            sched_params={
                'patience': 2,
                'min_lr': 1e-4},
            sched_ext_update=False,
            input_transform=lambda a: a):

        super(NetTrainer, self).__init__()

        # declare primary device
        self._state.device = torch.device(device)
        # relocate network
        if multi_gpu:
            assert device == 'cuda:0',\
                    'The primary device has to be \'cuda:0\' to enable multiple GPUs'
            net = torch.nn.DataParallel(net)
        self._state.net = net.to(self._state.device)
        # relocate loss funciton
        self._state.criterion = loss_fn.to(self._state.device)
        # initialize optimizer
        if optim == 'SGD':
            self._state.optimizer = torch.optim.SGD(self._state.net.parameters(),
                    **optim_params)
        elif optim == 'Adam':
            self._state.optimizer = torch.optim.Adam(self._state.net.paramters(),
                    **optim.params)
        else:
            raise ValueError('Unsupported optimizer type {}'.format(optim))
        # load optimzer state dict if provided
        if optim_state_dict is not None:
            self._state.optimizer.load_state_dict(optim_state_dict)

            """THE FOLLOWING RESOLVES AN ISSUE WITH USAGE OF GPU"""
            for state in self._state.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self._state.device)
        # initialize learning rate scheduler
        if lr_scheduler:
            self._state.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._state.optimizer,
                    **sched_params)
        else:
            self._state.scheduler = None
        # initilize logger
        if cache_dir is None:
            self._state.logger = None
        else:
            self._state.logger = TrainLog(os.path.join(cache_dir, 'train_log'))
        
        self._state.step = 0
        self._state.epoch = 0
        self._state.multi_gpu = multi_gpu
        self._state.cache_dir = cache_dir
        self._state.train_loader = train_loader
        self._state.print_interval = print_interval
        self._state.val_loader = val_loader
        self._state.sched_ext_update = sched_ext_update
        self._state.input_transform = input_transform
        if loss_fn_val is None:
            self._state.criterion_val = loss_fn
        else:
            self._state.criterion_val = loss_fn_val

        if cache_dir is not None and not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

    def __call__(self, nepoch):
        """Train the network"""
        self._state.net.train()

        for _ in range(nepoch):
            self._state.epoch += 1
            # accumulated validation loss
            running_loss = 0

            # print epoch stats
            if self._state.logger is not None:
                self._state.logger.write('\n\nEPOCH {:02d} INITIATED\n'.format(self._state.epoch))
                self._state.logger.write('Learning rate: {}, step: {}\n\n'.format(
                    self._state.optimizer.state_dict()['param_groups'][0]['lr'], self._state.step))

            for dtuple in self._state.train_loader:
                # format input data
                dtuple = self._state.input_transform(dtuple)
                # relocate tensors to designated device
                dtuple = [item.to(self._state.device) for item in dtuple]
                # forward pass
                out = self._state.net(*dtuple[:-1])
                # compute loss
                loss = self._state.criterion(out, dtuple[-1])
                self._state.optimizer.zero_grad()
                # back propogation
                loss.backward()
                # update weights
                self._state.optimizer.step()
                
                # run validation and print losses
                if self._state.step % self._state.print_interval == 0:
                    val_loss = self._val()
                    if val_loss is None:
                        # accumulate training loss if validation is disabled
                        running_loss += loss.item()
                    else:
                        running_loss += val_loss
                    if self._state.logger is not None:
                        self._state.logger.log(self._state.step, loss.item(), val_loss)

                self._state.step += 1
            
            # update learning rate
            if self._state.scheduler is not None and not self._state.sched_ext_update:
                self._state.scheduler.step(running_loss)
            # save checkpoint at the end of each epoch
            if self._state.cache_dir is not None:
                if self._state.multi_gpu:
                    net_copy = copy.deepcopy(self._state.net.module).cpu()
                else:
                    net_copy = copy.deepcopy(self._state.net).cpu()
                # copy the optimizer
                optim_copy = copy.deepcopy(self._state.optimizer)
                for state in optim_copy.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()
                torch.save({
                    'step': self._state.step,
                    'epoch': self._state.epoch,
                    'model_state_dict': net_copy.state_dict(),
                    'optim_state_dict': optim_copy.state_dict()
                    }, os.path.join(self._state.cache_dir, 'torch_ckpt_s{:05d}_e{:02d}'.\
                            format(self._state.step, self._state.epoch)))

    def _val(self):
        """Run validation"""
        # sample one minibatch of validation data from a dataloader
        if self._state.val_loader is not None:
            try:
                dtuple = next(val_iter)
            except (NameError, StopIteration):
                val_iter = iter(self._state.val_loader)
                dtuple = next(val_iter)
            dtuple = self._state.input_transform(dtuple)
            dtuple = [item.float().to(self._state.device) for item in dtuple]
            with torch.no_grad():
                if len(dtuple) == 2:
                    out = self._state.net(dtuple[0])
                else:
                    out = self._state.net(*dtuple[:-1])
            loss = self._state.criterion_val(out, dtuple[-1])
            return loss.item()
        # return None when no validation data is provided
        else:
            return None

class InferenceManager(Engine):
    """
    Perform inference on an entire dataloader

    Arguments:

    [REQUIRED ARGS]
        net(Module): Network model
        dataloader(iterable): Dataloader with batch data as [INPUT_1, ..., INPUT_N],
            where each input should be a Tensor. Otherwise, use {input_transform}
            to format the batch

    [OPTIONAL ARGS]
        device(str): Primary device, has to be cuda:0 when using multiple gpus
        multi_gpu(bool): When True, use all visible gpus for forward pass
        cache_dir(str): Diretory to save cache
        save_output(bool): When True, save the output to a pickle file
        print_interval(int): Number of batches between two progress updates
        input_transform(callable): Transform the batch data to desired format
        output_transform(callable): Transform the collective output to desired format
            By default, the collective output is [BATCH_1_OUTPUT, ..., BATCH_N_OUTPUT]

    Example:

        >>> import torch
        >>> from pocket.utils import InferenceManager
        >>> net = torch.nn.Linear(1, 1, bias=False)
        >>> net.weight.data = torch.tensor([[2.]])
        >>> dataloader = [[torch.tensor([[1.]])], [torch.tensor([[2.]])]]
        >>> a = InferenceManager(net, dataloader)
        >>> a()
        >>> [tensor([[2.]]), tensor([[4.]])]
    """
    def __init__(self,
            net,
            dataloader,
            device='cpu',
            cache_dir=None,
            multi_gpu=False,
            save_output=False,
            print_interval=100,
            input_transform=lambda a: a,
            output_transform=lambda a: a):

        super(InferenceManager, self).__init__()

        if save_output == True and cache_dir is None:
            raise ValueError('When save_output is True, cache_dir has to be specified')

        # set primary device
        self._state.device = torch.device(device)
        if multi_gpu:
            assert device == 'cuda:0',\
                    'The primary device has to be cuda:0 for multi-gpu usage'
            net = torch.nn.DataParallel(net)
        self._state.net = net.to(self._state.device)

        if cache_dir is None:
            self._state.logger = None
        else:
            self._state.logger = Log(os.path.join(cache_dir, 'forward_log'), 'a')

        self._state.cache_dir = cache_dir
        self._state.dataloader = dataloader
        self._state.save_output = save_output
        self._state.print_interval = print_interval
        self._state.input_transform = input_transform
        self._state.output_transform = output_transform

        if cache_dir is not None and not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

    def __call__(self):
        """
        Returns:
            output(ndarray): Each element would be the exact output of the
                corresponding minibatch
        """
        output = []
        
        if self._state.logger is not None:
            self._state.logger.write('\nSTART FORWARD PASS\n')
        for i, dtuple in enumerate(self._state.dataloader):
            # apply transform to the input data
            dtuple = self._state.input_transform(dtuple)
            # relocate tensor to primary device
            dtuple = [item.to(self._state.device) for item in dtuple]
            # forward pass
            with torch.no_grad():
                out = self._state.net(*dtuple).cpu()
            # store the batch output
            output.append(out)

            if self._state.logger is not None and i % self._state.print_interval == 0:
                self._state.logger.write('\nBATCH {} FINISHED\n'.format(i))
    
        output = self._state.output_transform(output)
        if self._state.save_output:
            self._state.logger.write('\nSAVING OUTPUT TO CACHE...\n')
            with open(os.path.join(self._state.cache_dir, 'output.pkl'), 'wb') as f:
                pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
            self._state.logger.write('\nDONE\n')
        
        return output
