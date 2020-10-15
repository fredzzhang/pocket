"""
Dataset base classes

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import pickle
from PIL import Image
from torch.utils.data import Dataset

class DataDict(dict):
    r"""
    Data dictionary class. This is a class based on python dict, with
    augmented utility for loading and saving
    
    Arguments:
        input_dict(dict, optional): A Python dictionary
        kwargs: Keyworded arguments to be stored in the dict

    Example:

        >>> from pocket.data import DataDict
        >>> person = DataDict()
        >>> person.is_empty()
        True
        >>> person.age = 15
        >>> person.sex = 'male'
        >>> person.save('./person.pkl', 'w')
    """
    def __init__(self, input_dict=None, **kwargs):
        data_dict = dict() if input_dict is None else input_dict
        data_dict.update(kwargs)
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


# Define identity mapping at the top level to make it picklable
def I(x):
    return x
def I2(x, y):
    return x, y

class ImageDataset(Dataset):
    """
    Base class for image dataset

    Arguments:
        root(str): Root directory where images are downloaded to
        transform(callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version.
    """
    def __init__(self, root, transform=None, target_transform=None, transforms=None):
        self._root = root
        self._transform = I if transform is None else transform
        self._target_transform = I if target_transform is None else target_transform
        self._transforms = I2 if transforms is None else transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError
    
    def __repr__(self):
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=\"' + repr(self._root)
        reprstr += '\")'
        # Ignore the optional arguments
        return reprstr

    def __str__(self):
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tRoot path: {}\n'.format(self._root)
        return reprstr

    def load_image(self, path):
        """Load an image and apply transform"""
        return Image.open(path)
