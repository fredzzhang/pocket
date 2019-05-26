"""
Classes used for visualization purposes

Written by Frederic Zhang
Australian National University

Last updated in Apr. 2019
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

class LearningCurveVisualizer:
    """
    Plot learning curve

    Arguments:
        src_path(str): path of source .txt file

    Assume the valid lines in the .txt file has the following format
        1. Losses Type I
            Step: {A}, training loss: {B}, validation loss: {C}
        2. Losses Type II
            Step: {A}, training loss: {B}
        3. Learning Rate
            Learning rate: {A}, step: {B}

    Othewise lines will be disregarded
    """
    def __init__(self, src_path):
        step = []
        train_loss = []
        val_loss = []

        estep = []
        lr = []
        f = open(src_path, 'r')
        for line in f:
            if line.startswith('Step'):
                seg = line.split(',')
                assert len(seg) in [2, 3],\
                        'For losses, there should be 2 or 3 segments' +\
                        'in a valid line seperated by commas'
                step.append(int(seg[0].split()[-1]))
                train_loss.append(float(seg[1].split()[-1]))
                if len(seg) == 3:
                    val_loss.append(float(seg[2].split()[-1]))
            elif line.startswith('Learning'):
                seg = line.split(',')
                assert len(seg) == 2,\
                        'For learning rate, there should be 2 segments' +\
                        'in a valid line seperated by comma'
                lr.append(float(seg[0].split()[-1]))
                estep.append(int(seg[1].split()[-1]))

        self._step = np.asarray(step)
        self._train_loss = np.asarray(train_loss)
        if len(val_loss):
            self._val_loss = np.asarray(val_loss)
        else:
            self._val_loss = None
        if len(lr):
            self._lr = np.asarray(lr)
            self._estep = np.asarray(estep)
        else:
            self._lr = None
            self._estep = None

    def _smooth(self, weight):
        """
        1st-order IIR low-pass filter to smooth out the learning curves

        Arguments:
            weight(float): smoothing factor between 0 and 1
        """
        assert weight >=0 and weight <= 1,\
                'Invalid smoothing factor {:2f}. Choose between 0 and 1'.format(weight)
        train_loss = np.zeros_like(self._train_loss)
        for i in range(len(self._train_loss)):
            if i == 0:
                train_loss[i] = self._train_loss[i]
            train_loss[i] = self._train_loss[i - 1] * weight\
                    + (1 - weight) * self._train_loss[i]

        val_loss = np.zeros_like(self._val_loss)
        if self._val_loss is not None:
            for i in range(len(self._val_loss)):
                if i == 0:
                    val_loss[i] = self._val_loss[i]
                val_loss[i] = self._val_loss[i - 1] * weight\
                        + (1 - weight) * self._val_loss[i]

        return train_loss, val_loss

    @property
    def step(self):
        return self._step

    @property
    def loss(self):
        return self._train_loss, self._val_loss

    def show(self, scale='linear', weight=0):
        """Plot the learning curves"""
        if weight == 0:
            train_loss, val_loss = self._train_loss, self._val_loss
        else:
            train_loss, val_loss = self._smooth(weight)

        fig, ax1 = plt.subplots()
        fig.set_figwidth(10)
        fig.set_figheight(5)

        seg = ax1.plot(self._step, train_loss, '-b', label='Training Loss', alpha=.8)
        if self._val_loss is not None:
            seg += ax1.plot(self._step, val_loss, '-r', label='Validation Loss', alpha=.4)
        ax1.set_ylabel('Loss')
        ax1.set_yscale(scale)
        ax1.grid()

        if self._lr is not None:
            ax2 = ax1.twinx()
            seg += ax2.plot(self._estep, self._lr, '-gd', label='Learning Rate', alpha=.4)
            ax2.set_ylabel('Learning rate')
            ax2.set_xlabel('Step')
            ax2.set_yscale('log')
            # merge legend
            labels = [l.get_label() for l in seg]
            ax2.legend(seg, labels)
        else:
            ax1.legend()

        plt.title('Learning Curves')
        plt.show()
    
class ParamVisualizer:
    """
    Parameter visualizer

    Arguments:
        params(torch.tensor): weights or biases of torch.nn modules

    Currently supports F.C. and Conv. layers
    """
    def __init__(self, params):
        if type(params) == torch.Tensor:
            self._params = params.cpu().numpy()
        elif type(params) == np.ndarray:
            self._params = params
        else:
            raise TypeError('Unsupported data type {}'.format(type(params)))
        self._plotter = {
                1: self._plot_biases,
                2: self._plot_fc_weights,
                4: self._plot_conv_weights}
    
    @property
    def params(self):
        """Return the parameters"""
        return self._params

    def show(self):
        """
        Plot the parameters according to the dimension
            (OUT, IN): weights of F.C. layers
            (OUT): biases of F.C. or Conv. layers
            (OUT, IN, K, K): weights of Conv. layers
        """
        self._plotter[len(self._params.shape)]()

    def _plot_fc_weights(self):
        """Plot the weights of an F.C. layer"""
        H, W = self._params.shape
        gain = 10.0 / np.maximum(H, W)
        plt.figure(figsize=(W * gain, H * gain))
        plt.imshow(self._params, cmap='hot')
        plt.colorbar()
        plt.xlabel('Input Nodes')
        plt.ylabel('Output Nodes')
        plt.title('Weights of an F.C. layer')
        plt.show()

    def _plot_conv_weights(self):
        """Plot the weights of a Conv. layer"""
        raise NotImplementedError

    def _plot_biases(self):
        """Plot the biases"""
        plt.figure(figsize=(10, 4))
        x = np.linspace(0, len(self._params) - 1, len(self._params))
        plt.plot(x, self._params, 'g.', x, self._params)
        plt.xlabel('Output Nodes')
        plt.title('Biases')
        plt.show()

class NetVisualizer(dict):
    """
    Network parameter visualizer

    Arguments:
        pt_path(str): path of a PyTorch model, typically ends with .pt
        ckpt_path(str): path of a checkpoint file, with field 'model_state_dict'
    """
    def __init__(self, pt_path=None, ckpt_path=None):
        if pt_path is not None:
            self._param_dict = torch.load(pt_path)
        elif ckpt_path is not None:
            self._param_dict = torch.load(ckpt_path)['model_state_dict']
        else:
            raise ValueError('No valid path given')
        for key in self._param_dict:
            self[key] = ParamVisualizer(self._param_dict[key])

    def __len__(self):
        """Return the number of parameter blocks"""
        return len(self._param_dict)

    @property
    def keys(self):
        """Return the name of parameter blocks"""
        return self._param_dict.keys()

class ImageVisualizer:
    """
    Image visualizer

    Arguments:
        imdir(str): directory of source images
        labels(ndarray): (N, M) one-hot ecoded labels for all N images
            in the directory, across M classes
        descpt(ndarray): (M,) string descriptons for each of M classes 
    """
    def __init__(self, imdir, labels=None, descpt=None):
        # get the image paths in alphabetic order
        self._construct_image_paths(imdir)
        if labels is not None:
            assert labels.shape[0] == len(self._im_paths), \
                    'Not enough labels ({}) for all images ({})'.\
                    format(labels.shape[0], len(self._im_paths))
            if descpt is not None:
                assert len(descpt) == labels.shape[1], \
                        'Length of descrption ({}) does not match the number of labels ({})'.\
                        format(len(descpt), labels.shape[1])

        self._labels = labels
        self._descpt = descpt

    def __len__(self):
        """Return the number fo images the visualizer can show"""
        return len(self._im_paths)

    def _construct_image_paths(self, imdir):
        """Find all images with extension .jpg or .png"""
        self._im_paths = [os.path.join(imdir, f) for f in os.listdir(imdir) \
                if f.endswith('.jpg') or f.endswith('.png')]  
        self._im_paths.sort()

    def show(self, i):
        """Display the designatd image and corresponding labels"""
        im = cv2.imread(self._im_paths[i])
        if self._labels is not None:
            im_label = np.where(self._labels[i, :])[0]
            assert len(im_label) != 0, \
                    'Selected image does not have a positive label'
            assert len(im_label) == 1, \
                    'Multi-label is not supported at the moment'
            if self._descpt is not None:
                cv2.imshow(self._descpt[im_label].item(), im)
            else:
                cv2.imshow('Class {}'.format(im_label[0]), im)
            cv2.waitKey(1)

    def image_path(self, i):
        """Return the path for image corresponding to index i"""
        return self._im_paths[i]

class BoxVisualizer(ImageVisualizer):
    """
    Bounding box visualizer

    Arguments:
        imdir(str): directory of source images
        boxdir(str): directory of bounding box files

    Assume the coordinates of bounding boxes are stored in .txt files
    """
    def __init__(self, imdir, boxdir):
        super(BoxVisualizer, self).__init__(imdir)
        self._construct_box_paths(boxdir)

    def _construct_box_paths(self, boxdir):
        """Find all .txt files"""
        self._box_paths = [os.path.join(boxdir, f) for f in os.listdir(boxdir) \
                if f.endswith('.txt')]
        self._box_paths.sort()

    def show(self, i):
        """Display all bounding boxes in an image"""
        raise NotImplementedError

