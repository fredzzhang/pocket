"""
Train and test a neural network
under the PyTorch framework

Written by Frederic Zhang
Australian National University

Last updated in Apr. 2019
"""

import os
import copy
import torch
import pickle
import numpy as np
from heapq import heappush, heappop

from utils.io import Log, TrainLog
from utils.cython_bbox import bbox_overlaps

class NetTrainer:
    """
    Network trainer class

    Arguments:
    
    [REQUIRED ARGS]
        net(Module): the network to be trained
        loss_fn(Module): loss function handle
        cache_dir(str): diretory to saved models and training log
        train_loader(iterable): dataloader for training set, preferably as DataLoader type
            batch output should have the format [INPUT_1, ..., INPUT_N, LABELS]
        print_interval(int): number of steps to print training and validation losses once

    [OPTIONAL ARGS]
        optim(str): optimizer used in training, 'SGD', 'Adam'
        optim_params(dict): optimizer parameter dict with same keywords as selected optimizer
            default: {'lr':0.1, 'momentum':0.9, 'weight_decay':5e-4}
        optim_state_dict(OrderedDict): state dict of the optimizer
        device(str): the primary device, e.g. 'cpu', 'cuda:0'
        multigpu(bool): allows multi-GPU computation. When being true, {device} has to be 'cuda:0'
        val_data(tuple): a tuple of validation data [INPUT_1, ..., INPUT_N, LABELS]
        val_loader(iterable): dataloader for validation set with the same data format as train loader
        lr_scheduler(bool): use learning rate scheduler, default: True
        sched_params(dict): learning rate scheduler parameter dict
        formatter(handle): handle of a function to format the data tuple extracted from dataloader
            default: lambda a: a -> keep the original format 
    """
    def __init__(self,
            net,
            loss_fn,
            cache_dir,
            train_loader,
            print_interval,
            optim='SGD',
            optim_params={
                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4},
            optim_state_dict=None,
            device='cpu',
            multigpu=False,
            val_data=None,
            val_loader=None,
            loss_fn_val=None,
            lr_scheduler=True,
            sched_params={
                'patience': 2,
                'min_lr': 1e-4},
            formatter=lambda a: a):

        # declare primary device
        self._device = torch.device(device)
        # relocate network
        if multigpu:
            assert device == 'cuda:0',\
                    'The primary device has to be \'cuda:0\' to enable multiple GPUs'
            net = torch.nn.DataParallel(net)
        self._net = net.train().to(self._device)
        # initialize optimizer
        if optim == 'SGD':
            self._optimizer = torch.optim.SGD(self._net.parameters(),
                    **optim_params)
        elif optim == 'Adam':
            self._optimizer = torch.optim.Adam(self._net.paramters(),
                    **optim.params)
        else:
            raise ValueError
        # load optimzer state dict if provided
        if optim_state_dict is not None:
            self._optimizer.load_state_dict(optim_state_dict)

            """THE FOLLOWING RESOLVES AN ISSUE WITH USAGE OF GPU"""
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self._device)
        # initialize learning rate scheduler
        if lr_scheduler:
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._optimizer,
                    **sched_params)
        else:
            self._scheduler = None
        # reformat and relocate validation data if provided
        if val_data is not None:
            val_data = formatter(val_data)
            self._val_data = [item.float().to(self._device) for item in val_data]
        else:
            self._val_data = None
        # initilize logger
        self._logger = TrainLog(os.path.join(cache_dir, 'train_log'))
        
        self._step = 0
        self._epoch = 0
        self._loss_fn = loss_fn
        self._multigpu = multigpu
        self._cache_dir = cache_dir
        self._train_loader = train_loader
        self._print_interval = print_interval
        self._val_loader = val_loader
        self._formatter = formatter
        if loss_fn_val is None:
            self._loss_fn_val = loss_fn
        else:
            self._loss_fn_val = loss_fn_val

    @property
    def step(self):
        """Current step number"""
        return self._step

    @step.setter
    def step(self, n):
        """Change current step number"""
        if type(n) == int:
            self._step = n
        else:
            raise ValueError('The step number has to be of int type')

    @property
    def epoch(self):
        """Current epoch number"""
        return self._epoch

    @epoch.setter
    def epoch(self, n):
        """Change current epoch number"""
        if type(n) == int:
            self._epoch = n
        else:
            raise ValueError('The epoch number has to be of int type')

    def train(self, nepoch):
        """Train the network"""
        self._logger.write('\n\nTRAINING INITIATED\n')
        self._logger.write('Learning rate: {}, step: {}\n\n'.format(
            self._optimizer.state_dict()['param_groups'][0]['lr'], self._step))

        for _ in range(nepoch):
            self._epoch += 1
            # accumulated validation loss
            running_loss = 0

            for dtuple in self._train_loader:
                # use wrapper to format the data
                dtuple = self._formatter(dtuple)
                # relocate tensors to designated device
                dtuple = [item.float().to(self._device) for item in dtuple]
                # forward pass
                if len(dtuple) == 2:
                    out = self._net(dtuple[0])
                else:
                    out = self._net(*dtuple[:-1])
                # compute loss
                loss = self._loss_fn(out, dtuple[-1])
                self._optimizer.zero_grad()
                # back propogation
                loss.backward()
                # update weights
                self._optimizer.step()
                
                # run validation and print losses
                if self._step % self._print_interval == 0:
                    val_loss = self._val()
                    running_loss += val_loss
                    self._print_stats(loss.item(), val_loss)

                self._step += 1
            
            # update learning rate
            if self._scheduler is not None:
                self._scheduler.step(running_loss)
            # print epoch stats
            self._logger.write('\n\nEPOCH {:02d} FINISHED\n'.format(self._epoch))
            self._logger.write('Learning rate: {}, step: {}\n\n'.format(
                self._optimizer.state_dict()['param_groups'][0]['lr'], self._step))
            # save checkpoint at the end of each epoch
            if self._multigpu:
                net_copy = copy.deepcopy(self._net.module).cpu()
            else:
                net_copy = copy.deepcopy(self._net).cpu()
            # copy the optimizer
            optim_copy = copy.deepcopy(self._optimizer)
            for state in optim_copy.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()
            torch.save({
                'step': self._step,
                'epoch': self._epoch,
                'model_state_dict': net_copy.state_dict(),
                'optim_state_dict': optim_copy.state_dict()
                }, os.path.join(self._cache_dir, 'torch_ckpt_s{:05d}_e{:02d}'.\
                        format(self._step, self._epoch)))
            # reset the statistics of loss function
            try:    self._loss_fn.reset()
            except: pass

    def _val(self):
        """Run validation"""
        # use constant validation data if provided
        if self._val_data is not None:
            with torch.no_grad():
                if len(self._val_data) == 2:
                    out = self._net(self._val_data[0])
                else:
                    out = self._net(*self._val_data[:-1])
            loss = self._loss_fn_val(out, self._val_data[-1])
            return loss.item()
        # sample one minibatch of validation data from a dataloader
        elif self._val_loader is not None:
            try:
                dtuple = next(val_iter)
            except (NameError, StopIteration):
                val_iter = iter(self._val_loader)
                dtuple = next(val_iter)
            dtuple = self._formatter(dtuple)
            dtuple = [item.float().to(self._device) for item in dtuple]
            with torch.no_grad():
                if len(dtuple) == 2:
                    out = self._net(dtuple[0])
                else:
                    out = self._net(*dtuple[:-1])
            loss = self._loss_fn_val(out, dtuple[-1])
            return loss.item()
        # return None when no validation data is provided
        else:
            return None

    def _print_stats(self, train_loss, val_loss):
        """Log training and validation losses"""
        self._logger.log(self._step, train_loss, val_loss)


class NetTester:
    """
    Network tester class

    Arguments:

    [REQUIRED ARGS]
        net(Module): the network to be tested
        data_loader(iterable): dataloader for test set, preferably as DataLoader type
            by default, batch output should have the format [INPUT_1, ..., INPUT_N, LABELS]
            if otherwise, use {formatter} to format the input data
        
        NOTE: for mAP setting, the format should be [INPUT_1, ..., INPUT_N, BOX_COORDS]
        and data have to been organized on a per-image basis

    [OPTIONAL ARGS]
        device(str): the device to be used, e.g. 'cpu', 'cuda:0'
        print_interval(int): number of steps to log progress
        cache_dir(str): cache directory for test log
        formatter(handle): handle of a function to format the data tuple extracted from dataloader
            default: lambda a: a -> keep the original format
    """
    def __init__(self,
            net,
            data_loader,
            device='cpu',
            print_interval=500,
            cache_dir='./cache',
            formatter=lambda a: a):
        
        # declare primary device
        self._device = torch.device(device)
        # initialize logger
        self._logger = Log(os.path.join(cache_dir, 'test_log'), 'a')

        self._eval_metric = None
        self._eval_method = NotImplemented
        self._forward_method = NotImplemented
        self._net = net
        self._cache_dir = cache_dir
        self._data_loader = data_loader
        self._print_interval = print_interval
        self._formatter = formatter

    @property
    def metric(self):
        """Evaluation metric"""
        return self._eval_metric

    @property
    def gtdb(self):
        """Ground truth database if exists"""
        try:
            return self._gtdb
        except:
            raise NotImplementedError

    @property
    def detdb(self):
        """Detection database if exists"""
        try:
            return self._detdb
        except:
            raise NotImplementedError

    def eval(self):
        """Evaluate based on the specified metric"""
        if self._detdb is None:
            assert self._net is not None,\
                    'Please pass a valid torch.nn.Module'
            assert self._data_loader is not None,\
                    'Dataloader is None object'
            self._logger.write('\nSTART FORWARD PASS...\n\n')
            self._forward_method()
            self._logger.write('\nFOWARD PASS COMPLETED\n')

        assert self._detdb.shape == (self._num_images, self._num_classes),\
                'Invalid size {} for detection database'.format(self._detdb.shape)
        self._logger.write('\nSTART EVALUATION...\n\n')
        self._eval_method()

    def set_eval_metric(self, metric, **kwargs):
        """
        Set evaluation metric

        Arguments:
            metric(str): evaluation metric, only 'mAP' is supported for now

            kwargs will vary based on the metric
            'mAP'
                gt_dir(str): the directory for ground truth files
                min_IoU(float): minimum IoU for data association
                num_classes(int): number of target classes
                samples_per_class(int): number of samples kept per target class
                detdb(Array of Objects): detection results arranged in (N, M) array
                    of objects, where N is the number of images and M is the number
                    of classes. Each entry should be a ndarray of box pairs in the
                    following format
                        [H_x1, H_y1, H_x2, H_y2, O_x1, O_y1, O_x2, O_y2, Score]
        """
        self._eval_metric = metric    
        if metric == 'mAP':
            self._prepare_for_mAP(**kwargs)
            self._eval_method = self._eval_mAP
            self._forward_method = self._forward_mAP
        else:
            raise NotImplementedError

    def _prepare_for_mAP(self, 
            gt_dir,
            min_IoU,
            num_classes,
            samples_per_class,
            detdb=None):
        """Prepare ground truth bounding boxes for evaluating mAP"""

        self._min_IoU = min_IoU
        self._num_classes = num_classes
        self._samples_per_class = samples_per_class
        self._detdb = detdb
        self._empty = []

        gt_files = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir)\
                if f.endswith('.txt')]
        gt_files.sort()

        self._num_images = len(gt_files)
        # construct ground truth matrix (num_images, num_classes)
        self._gtdb = np.empty((self._num_images, self._num_classes),
                dtype=object)
        self._num_anno = np.zeros(self._num_classes)
        for i, f in enumerate(gt_files):
            boxes = np.loadtxt(f, ndmin=2)
            if boxes.shape[0] is 0:
                self._empty.append(i)
                continue
            unique_cls = np.unique(boxes[:, 0]).astype(np.int32)
            for j in unique_cls:
                inds = np.where(boxes[:, 0] == j)[0]
                self._gtdb[i, j] = boxes[inds, 1:]
                self._num_anno[j] += np.sum(boxes[:, 0] == j)

    def _eval_mAP(self):
        """Evaluate the mAP"""
        counter = np.zeros(self._num_classes, dtype=np.int32)
        tp = np.zeros((self._num_classes, self._samples_per_class))
        scores = np.zeros_like(tp)

        for i in range(self._num_images):
            # skip empty images
            if i in self._empty:
                continue
            for j in range(self._num_classes):
                # skip when there are no detection box pairs
                if self._detdb[i, j] is None:
                    continue
                if self._detdb[i, j].shape[0] * self._detdb[i, j].shape[1] == 0:
                    continue
                num_pairs = self._detdb[i, j].shape[0]
                # associate detection with ground truth if any
                if self._gtdb[i, j] is not None:
                    ov = np.minimum(
                            bbox_overlaps(
                                self._detdb[i, j][:, :4].astype(np.float),
                                self._gtdb[i, j][:, :4].astype(np.float)),
                            bbox_overlaps(
                                self._detdb[i, j][:, 4:8].astype(np.float),
                                self._gtdb[i, j][:, 4:].astype(np.float)))
                    match = ov > self._min_IoU
                    # now find out which G.T. box pairs are matched more than once
                    inds = np.where(np.sum(match, 0) > 1)[0]
                    for k in inds:
                        # regard the detection box pair with highest classifcation score as a match
                        det_inds = np.where(match[:, k])[0]
                        det_max = np.argmax(self._detdb[i, j][det_inds, -1])
                        vec = np.zeros_like(match[:, k])
                        vec[det_inds[det_max]] = 1
                        match[:, k] = vec
                    # sanity check
                    assert np.sum(np.sum(match, 0) > 1) == 0,\
                        'Ground truth box pairs matched more than once'
                    # mark true positives
                    tp[j, counter[j]: counter[j] + num_pairs] = np.sum(match, 1)

                # save the classification scores
                scores[j, counter[j]: counter[j] + num_pairs] = self._detdb[i, j][:, -1]
                counter[j] += num_pairs

        # sort the samples by classification scores in descending order
        perm = np.argsort(scores, 1)[:, ::-1]
        for i in range(perm.shape[0]):
            tp[i, :] = tp[i, perm[i, :]]

        self._ap = np.zeros(self._num_classes)
        fp = np.ones_like(tp) - tp.astype(np.bool)
        tp = np.cumsum(tp, 1)
        fp = np.cumsum(fp, 1)
        prec = tp / (tp + fp)
        rec = tp / self._num_anno[:, np.newaxis]
        # compute ap
        for j in range(self._num_classes):
            for t in np.linspace(0, 1, 11):
                # use 11-point interpolation to compute AP
                try:
                    p = np.max(prec[j, rec[j, :] >= t])
                    self._ap[j] += p / 11
                # ignore it when a certain recall is not reached
                except:
                    pass

        # save the data
        with open(os.path.join(self._cache_dir, 'eval.pkl'), 'wb') as f:
            pickle.dump({
                'tp': tp,
                'fp': fp,
                'prec': prec,
                'rec': rec,
                'ap': self._ap},
                f, pickle.HIGHEST_PROTOCOL)
        
        self._logger.time()
        self._logger.write('\nFor {} classes, mAP: {}, mRec: {}\n'.\
                format(self._num_classes, np.mean(self._ap), np.mean(rec[:, -1])))

    def _forward_mAP(self):
        """Run the network on the given dataset"""
        # prepare heaps
        top_scores = [[] for _ in range(self._num_classes)]
        # construct detections in the same format as ground truth
        self._detdb = np.empty((self._num_images, self._num_classes),
                dtype=object)
        # relocate device
        self._net = self._net.eval().to(self._device)

        for i, dtuple in enumerate(self._data_loader):
            # skip empty images
            if i in self._empty:
                continue
            # skip images without detections
            if len(dtuple[0]) == 0:
                continue
            
            # use wrapper to format the data
            dtuple = self._formatter(dtuple)
            # get bounding box pairs
            box_pairs = dtuple[-1].numpy()
            # relocate tensors to designated device
            dtuple = [item.float().to(self._device) for item in dtuple[:-1]]
            # forward pass
            with torch.no_grad():
                if len(dtuple) == 1:
                    out = self._net(dtuple[0]).cpu().numpy()
                else:
                    out = self._net(*dtuple).cpu().numpy()

            # update the heaps
            for j in range(self._num_classes):
                # save box pair cooords. and classificaiton scores
                self._detdb[i, j] = np.hstack((box_pairs, out[:, j: j+1]))
                # push the classification scores 
                for score in out[:, j]:
                    heappush(top_scores[j], score)
                # pop lowest scores to preserve maximum length
                while(len(top_scores[j]) > self._samples_per_class):
                    heappop(top_scores[j])

            if i % self._print_interval == 0:
                self._logger.time()
                self._logger.write('\nImage {} done\n'.format(i))

        # now use the lowest score in the heap as threshold
        # prune the box pairs
        for i in range(self._num_images):
            # skip empty images
            if i in self._empty:
                continue
            for j in range(self._num_classes):
                # skip images without detections
                if self._detdb[i, j] is None:
                    continue
                """
                HERE USE '>' INSTEAD OF '>=' TO AVOID POSSIBLE OVERFLOW,
                SINCE MULTIPLE SAMPLES COULD HAVE THE SAME CLASSIFICATION SCORE
                """
                inds = np.where(self._detdb[i, j][:, -1] > top_scores[j][0])[0]
                self._detdb[i, j] = self._detdb[i, j][inds, :]

        with open(os.path.join(self._cache_dir, 'detections.pkl'), 'wb') as f:
            pickle.dump(self._detdb, f, pickle.HIGHEST_PROTOCOL)
