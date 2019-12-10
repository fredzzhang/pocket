"""
HICODet dataset under PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os

from .base import ImageDataset

class HICODet(ImageDataset):
    """
    Arguments:
        root(str): Root directory where images are downloaded to
        annFile(str): Path to json annotation file
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
    """
    def __init__(self, root, annoFile, transform=None, target_transform=None):
        super(HICODet, self).__init__(root, annoFile, transform, target_transform)
        self._filenames = self._anno['filenames']
        self._class_corr = self._anno['class']
        self._empty_idx = self._anno['empty']
        self._anno = self._anno['annotation']

    def __len__(self):
        """Return the number of images"""
        return len(self._filenames)

    def __getitem__(self, i):
        """
        Arguments:
            i(int): Index to an image
        
        Returns:
            tuple[image, target]
        """
        return self.load_image(os.path.join(self._root, self._filenames[i])), \
            self._target_transform(self._anno[i])

    @staticmethod
    def load_annotation_and_metadata(f):
        return f['annotation'], f['filenames'], f['class'], f['empty']
        