"""
Compute the confusion matrix between multiple classes

Written by Frederic Zhang
Australian National Univeristy

Last updated in Jun. 2019
"""

import os
import torch
import pickle
import matplotlib.pyplot as plt

from ..utils import InferenceManager, load_pkl_from

class ConfusionMatrix:
    r"""
    Confusion matrix class, with dimension arranged in (Prediction, GroundTruth)
    Note that negative class, i.e. samples that don't belong to any classes, is 
    also included in the ground truth classes

    Arguments:
       num_cls(int, optional): Number of target classes in the matrix
       mode(string, optional): Evaluation mode, choose between 'FREQ' and 'MEAN'

    Example:
        
        >>> import torch
        >>> from pocket.diagnose import ConfusionMatrix
        >>> # Two samples belong to class 0, while both have higher scores for class 1
        >>> # One sample belongs to class 1, and is predicted with a higher score for class 1
        >>> # There are no negative samples
        >>> output = torch.tensor([[0.1, 0.9], [0.2, 0.7], [0.4, 0.9]])
        >>> labels = torch.tensor([[0., 1.], [1., 0.], [1., 0.]])
        >>> cm = ConfusionMatrix(2)
        >>> cm.push(output, labels)
        >>> cm.cmatrix
        tensor([[0., 0., 0.],
                [2., 1., 0.]])
        >>> cm = ConfusionMatrix(2, 'MEAN')
        >>> cm.push(output, labels)
        >>> cm.cmatrix
        tensor([[0.3000, 0.1000, 0.0000],
                [0.8000, 0.9000, 0.0000]])

    """
    def __init__(self, num_cls=0, mode='FREQ'):
        """Constructor method"""
        assert mode in ['FREQ', 'MEAN'],\
           'There are only two options for mode: \'FREQ\' and \'MEAN\''
        self._num_cls = num_cls
        self._mode = mode
        # Include negative class in ground truth
        self._cmatrix = torch.zeros(num_cls, num_cls + 1)
        if mode == 'MEAN':
            self._count = torch.zeros(num_cls + 1)

    def _update_freq(self, out, labels):
        """Update the frequency of predictions"""
        pred_cls = torch.argmax(out, 1)
        gt_cls = torch.nonzero(labels)
        for ind in gt_cls:
            self._cmatrix[pred_cls[ind[0]], ind[1]] += 1
        neg_samples = torch.nonzero(torch.sum(labels, 1) == 0)
        for ind in neg_samples:
            self._cmatrix[pred_cls[ind], -1] += 1
       
    def _update_mean(self, out, labels):
        """Update accumulated prediction scores"""
        gt_cls = torch.nonzero(labels)
        self._cmatrix *= self._count
        for ind in gt_cls:
            self._cmatrix[:, ind[1]] += out[ind[0], :]
            self._count[ind[1]] += 1
        neg_samples = torch.nonzero(torch.sum(labels, 1) == 0)
        for ind in neg_samples:
            self._cmatrix[:, -1] += out[ind[0], :]
            self._count[-1] += 1
        self._cmatrix /= self._count
        self._cmatrix[torch.isnan(self._cmatrix)] = 0

    @property
    def cmatrix(self):
        """The confusion matrix"""
        return self._cmatrix

    @property
    def mode(self):
        """The evaluation mode"""
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
        labels,
        dataloader,
        mode='FREQ',
        device='cpu',
        cache_dir='./',
        multi_gpu=False,
        input_transform=lambda a: a,
        output_transform=lambda a: torch.cat([b for b in a], 0)):
    """
    Compute the confusion matrix

    Arguments:

    [REQUIRED ARGS]
        net(Module): network model
        labels(Tensor[N, C]): Labels for all the data in loader, where N is the
            number of samples and c is the number of classes. NOTE: the order of
            the labels should be consistent with order of samples in dataloader
        dataloader(DataLoader): as the name suggests, 
            batch output should have the format [input_1, ..., input_N]

    [OPTIONAL ARGS]
        mode(string): choose between 'FREQ' and 'MEAN'
            'FREQ' - take the class with highest score as prediction and record
                the total number of predictions in the matrix
            'MEAN' - take the mean of output scores as entries in the matrix
        device(string): device to be used for network forward pass
            e.g. 'cpu', 'cuda:0'
        multi_gpu((bool): If True, use all visible GPUs during forward pass
        cache_dir(str): Directory to save cache
        input_transform(callable): Transform applied on batch data
        output_transform(callable): Transform applied on the collective output
    """
    if os.path.exists(os.path.join(cache_dir, 'output.pkl')):
        output = load_pkl_from(os.path.join(cache_dir, 'output.pkl'))
    else:
        inference = InferenceManager(
                net,
                dataloader,
                device,
                cache_dir,
                multi_gpu,
                True,
                100,
                input_transform)
        output = inference()
    output = output_transform(output)

    cmatrix = ConfusionMatrix(output.shape[1], mode)
    cmatrix.push(output, labels)

    return cmatrix
