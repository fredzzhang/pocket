"""
Dataset base classes

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import pickle
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Tuple

__all__ = ['DataDict', 'ImageDataset', 'DataSubset', 'DatasetConcat']

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
    def __init__(self, input_dict: Optional[dict] = None, **kwargs) -> None:
        data_dict = dict() if input_dict is None else input_dict
        data_dict.update(kwargs)
        super(DataDict, self).__init__(**data_dict)

    def __getattr__(self, name: str) -> Any:
        """Get attribute"""
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute"""
        self[name] = value

    def save(self, path: str, mode: str = 'wb', **kwargs) -> None:
        """Save the dict into a pickle file"""
        with open(path, mode) as f:
            pickle.dump(self.copy(), f, **kwargs)

    def load(self, path: str, mode: str = 'rb', **kwargs) -> None:
        """Load a dict or DataDict from pickle file"""
        with open(path, mode) as f:
            data_dict = pickle.load(f, **kwargs)
        for name in data_dict:
            self[name] = data_dict[name]

    def is_empty(self) -> bool:
        return not bool(len(self))

class StandardTransform:
    """https://github.com/pytorch/vision/blob/master/torchvision/datasets/vision.py"""

    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, inputs: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)

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
            and its target as entry and returns a transformed version
    """
    def __init__(self, root: str, transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None) -> None:
        self._root = root
        self._transform = transform
        self._target_transform = target_transform
        if transforms is None:
            self._transforms = StandardTransform(transform, target_transform)
        elif transform is not None or target_transform is not None:
            print("WARNING: Argument transforms is given, transform/target_transform are ignored.")
            self._transforms = transforms
        else:
            self._transforms = transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError
    
    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tRoot path: {}\n'.format(self._root)
        return reprstr

    def load_image(self, path: str) -> Image: 
        """Load an image as PIL.Image"""
        return Image.open(path).convert('RGB')

class DataSubset(Dataset):
    """
    A subset of data with access to all attributes of original dataset

    Arguments:
        dataset(Dataset): Original dataset
        pool(List[int]): The pool of indices for the subset
    """
    def __init__(self, dataset: Dataset, pool: List[int]) -> None:
        self.dataset = dataset
        self.pool = pool
    def __len__(self) -> int:
        return len(self.pool)
    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.pool[idx]]
    def __getattr__(self, key: str) -> Any:
        if hasattr(self.dataset, key):
            return getattr(self.dataset, key)
        else:
            raise AttributeError("Given dataset has no attribute \'{}\'".format(key))

class DatasetConcat(Dataset):
    """Combine multiple datasets into one

    Parameters:
    -----------
    args: List[Dataset]
        A list of datasets to be concatented
    """
    def __init__(self, *args: Dataset) -> None:
        self.datasets = args
        self.lengths = [len(dataset) for dataset in args]
    def __len__(self) -> int:
        return sum(self.lengths)
    def __getitem__(self, idx: int) -> Any:
        dataset_idx, intra_idx = self.compute_intra_idx(idx, self.lengths)
        return self.datasets[dataset_idx][intra_idx]
    @staticmethod
    def compute_intra_idx(idx: int, groups: List[int]) -> Tuple[int, int]:
        """Assume a sequence has been divided into multiple groups. Given
        a global index and the number of items each group has, find the
        corresponding group index and the intra index within the group

        Parameters:
        -----------
        idx: int
            Global index
        groups: List[int]
            Number of items in each group
        
        Returns:
        --------
        group_idx: int
            Index of the group
        intra_idx: int
            Intra index within the group
        """
        if idx >= sum(groups):
            raise ValueError(
                "The global index should be smaller "
                "than the length of the sequence."
            )
        groups = np.asarray(groups)
        cumsum = groups.cumsum()
        group_idx = np.where(idx < cumsum)[0].min()
        cumsum = np.concatenate([np.zeros(1, dtype=int), cumsum])
        intra_idx = idx - cumsum[group_idx]
        return group_idx, intra_idx