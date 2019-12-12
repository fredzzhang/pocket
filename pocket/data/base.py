"""
Dataset base classes

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import pickle
from PIL import Image
from torch.utils.data import Dataset

class DataDict(dict):
    r"""
    Data dictionary class. This is a class based on python dict, with
    augmented utility for loading and saving
    
    Arguments:
        data_dict(dict, optional): A Python dictionary

    Example:

        >>> from pocket.data import DataDict
        >>> person = DataDict()
        >>> person.is_empty()
        True
        >>> person.age = 15
        >>> person.sex = 'male'
        >>> person.save('./person.pkl', 'w')
    """
    def __init__(self, data_dict={}):
        """Constructor method"""
        super(DataDict, self).__init__(**data_dict)

    def __getattr__(self, name):
        """Get attribute"""
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        """Set attribute"""
        self[name] = value

    def save(self, path, mode='wb'):
        """Save the dict into a pickle file"""
        with open(path, mode) as f:
            pickle.dump(self.copy(), f, pickle.HIGHEST_PROTOCOL)

    def load(self, path, mode='rb'):
        """Load a dict or DataDict from pickle file"""
        with open(path, mode) as f:
            data_dict = pickle.load(f)
        for name in data_dict:
            self[name] = data_dict[name]

    def is_empty(self):
        return not bool(len(self))

class ImageDataset(Dataset):
    """
    Base class for image dataset

    Arguments:
        root(str): Root directory where images are downloaded to
        annFile(str): Path to json annotation file
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
    """
    def __init__(self, root, annoFile, transform=None, target_transform=None):
        self._root = root
        self._annoFile = annoFile
        with open(annoFile, 'r') as f:
            self._anno = json.load(f)
        self._transform = transform if transform is not None \
            else lambda a: a
        self._target_transform = target_transform if target_transform is not None \
            else lambda a: a

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError
    
    def __repr__(self):
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root) + ', '
        reprstr += 'annoFile='
        reprstr += repr(self._annoFile)
        reprstr += ', transform='
        reprstr += repr(self._transform)
        reprstr += ', target_transform='
        reprstr += repr(self._target_transform)
        reprstr += ')'
        return reprstr

    def __str__(self):
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tRoot path: {}\n'.format(self._root)
        return reprstr

    def load_image(self, path):
        """Load an image and apply transform"""
        return self._transform(Image.open(path))
