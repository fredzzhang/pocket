"""
Compute the confusion matrix between multiple classes

Written by Frederic Zhang
Australian National Univeristy

Last updated in May 2019
"""

import os
import torch
import pickle
import matplotlib.pyplot as plt

class ConfusionMatrix:
    """
    Confusion matrix class, with dimension arranged in (Prediction, GroundTruth)

    Arguments:
       num_cls(int): number of target classes in the matrix
       mode(string): evaluation mode, choose between 'FREQ' and 'MEAN'
    """
    def __init__(self, num_cls=0, mode='FREQ'):
        """Constructor method"""
        assert mode in ['FREQ', 'MEAN'],\
           'There are only two options for mode: \'FREQ\' and \'MEAN\''
        self._num_cls = num_cls
        self._mode = mode
        if mode == 'FREQ':
            self._cmatrix = torch.zeros(num_cls, num_cls + 1)
        else:
            self._count = torch.zeros(num_cls + 1)
            self._cmatrix = torch.zeros(num_cls, num_cls + 1)

    def _update_freq(self, out, labels):
        """Update the frequency of predictions"""
        pred_cls = torch.argmax(out, 1)
        gt_cls = torch.nonzero(labels)
        for ind in gt_cls:
            self._cmatrix[pred_cls[ind[0]], ind[1]] += 1
        neg_samples = torch.nonzero(torch.sum(labels, 1) == 0)[0]
        for ind in neg_samples:
            self._cmatrix[pred_cls[ind], -1] += 1
       
    def _update_mean(self, out, labels):
        """Update accumulated prediction scores"""
        gt_cls = torch.nonzero(labels)
        for ind in gt_cls:
            self._cmatrix[:, ind[1]] += out[ind[0], :]
            self._count[ind[1]] += 1
        neg_samples = torch.nonzero(torch.sum(labels, 1) == 0)
        for ind in neg_samples:
            self._cmatrix[:, -1] += out[ind[0], :]
            self._count[-1] += 1

    @property
    def cmatrix(self):
        """Return the confusion matrix"""
        return self._cmatrix

    @property
    def mode(self):
        """Return the evaluation mode"""
        return self._mode

    def reset(self):
        """Reset the confusion matrix with zeros"""
        self._cmatrix = torch.zeros_like(self._cmatrix)

    def push(self, out, labels):
        """
        Update the confusion matrix

        Args:
            out(Tensor): output of the network, (M, N)
            labels(Tensor): labels for the output, (M, N)
        """

        assert out.shape == labels.shape,\
            'Dimension of network output does not match that of the labels\
            \n{} != {}'.format(out.shape, labels.shape)
        assert self._num_cls == out.shape[1],\
            'Network does not have the same number of target classes,\
            \n{} != {}'.format(self._num_cls, out.shape[1])
        if self._mode == 'FREQ':
            self._update_freq(out, labels)
        else:
            self._update_mean(out, labels)

    def show(self):
        """
        Plot the confusion matrix

        In the plot, ground truth class is on the horizontal axis and predicted
        class is on the vertical axis
        """
        # widen the negative class for better visual clarity
        if self._num_cls > 150:
            cmatrix = torch.cat(
                    [self._cmatrix, 
                        torch.cat([torch.unsqueeze(self._cmatrix[:, -1], 1)] * 9, 1)], 1)
        else:
            cmatrix = self._cmatrix
        plt.figure(figsize=(10, 10))
        plt.imshow(cmatrix.numpy(), cmap='hot')
        plt.xlabel('Ground Truth Class')
        plt.ylabel('Predicted Class')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.show()

    def save(self, cache_dir):
        """Save the confusion matrix into a pickle file"""
        if self._mode == 'MEAN':
            self._cmatrix /= torch.cat(
                    [torch.unsqueeze(self._count, 0)] * self._num_cls, 0)
        with open(os.path.join(cache_dir, 'CMatrix_{}.pkl'.format(self._mode)), 'wb') as f:
            pickle.dump(self._cmatrix, f, pickle.HIGHEST_PROTOCOL)

    def load(self, cache_path):
        """Load the confuion matrix from a pickle file"""
        with open(cache_path, 'rb') as f:
            self._cmatrix = pickle.load(f)
        self._num_cls = self._cmatrix.shape[0]

    def merge(self, ind_range):
        """
        Merge certain classes of the confusion matrix

        Args:
            ind_range(Tensor): starting and end indices of intervals
                to be merged into one class, (N', 2)
        """
        self._num_cls = ind_range.shape[0]
        new_cmatrix = torch.zeros(self._num_cls, self._num_cls)
        # fill the new confusion matrix
        for i in range(self._num_cls):
            for j in range(self._num_cls):
                new_cmatrix[i, j] = torch.sum(self._cmatrix[ind_range[i, 0]: ind_range[i, 1] + 1,
                    ind_range[j, 0]: ind_range[j, 1] + 1])

        neg_cls = torch.zeros(self._num_cls, 1)
        for i in range(self._num_cls):
            neg_cls[i, 0] = torch.sum(self._cmatrix[ind_range[i, 0]: ind_range[i, 1] + 1, -1])
        new_cmatrix = torch.cat([new_cmatrix, neg_cls], 1)
        # update the confusion matrix
        self._cmatrix = new_cmatrix

    def normalize(self, dim=0):
        """Normalize the confusion matrix"""
        self._cmatrix /= torch.cat(
                [torch.sum(self._cmatrix, dim, keepdim=True)] * self._cmatrix.shape[dim],
                dim)

def compute_confusion_matrix(
        net,
        dataloader,
        cache_dir,
        class_of_interest,
        mode='FREQ',
        device='cpu',
        formatter=lambda a: a):
    """
    Arguments:

    [REQUIRED ARGS]
        net(Module): network model
        dataloader(DataLoader): as the name suggests, 
            batch output should have the format [input_1, ..., input_N, labels]
        cache_path(string): path to save the confusion matrix
        class_of_interest(Tensor): indices of classes in confusion matrix

    [OPTIONAL ARGS]
        mode(string): choose between 'FREQ' and 'MEAN'
            'FREQ' - take the class with highest score as prediction and record
                the total number of predictions in the matrix
            'MEAN' - take the mean of output scores as entries in the matrix
        device(string): device to be used for network forward pass
            e.g. 'cpu', 'cuda:0'
        formatter(function): handle of input data formater
            default: lambda a:a -> keep the original format
    """
    
    # declare primary device
    dev = torch.device(device)
    net = net.to(dev)
    # initialize the confusion matrix
    coi = class_of_interest.long()
    cmatrix = ConfusionMatrix(len(coi), mode)

    for dtuple in dataloader:
        # use data wrapper to format the input
        dtuple = formatter(dtuple)
        labels = dtuple[-1]
        # relocate tensor to designated device
        dtuple = [item.float().to(dev) for item in dtuple[:-1]]
        # forward pass
        with torch.no_grad():
            if len(dtuple) == 1:
                out = net(dtuple[0]).cpu()
            else:
                out = net(*dtuple).cpu()
        # update the confusion matrix
        cmatrix.push(out[:, coi], labels[:, coi])

    # save the matrix
    cmatrix.save(cache_dir)

    return cmatrix
