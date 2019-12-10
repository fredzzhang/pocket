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
        self._idx, self._anno, self._filenames, self._class_corr, self._empty_idx = \
            self.load_annotation_and_metadata(self._anno)

    def __len__(self):
        """Return the number of images"""
        return len(self._idx)

    def __getitem__(self, i):
        """
        Arguments:
            i(int): Index to an image
        
        Returns:
            tuple[image, target]
        """
        intra_idx = self._idx[i]
        return self.load_image(os.path.join(self._root, self._filenames[intra_idx])), \
            self._target_transform(self._anno[intra_idx])

    @property
    def class_corr(self):
        """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]
        """
        return self._class_corr.copy()

    @staticmethod
    def load_annotation_and_metadata(f):
        """
        Arguments:
            f(dict): Dictionary loaded from {annoFile}.json

        Returns:
            list[int]: Indices of images with valid interaction instances
            list[dict]: Annotations including bounding box pair coordinates and class index
            list[str]: File names for images
            list[list]: Class index correspondence
            list[int]: Indices of images without valid interation instances
        """
        idx = list(range(len(f['filenames'])))
        for empty_idx in f['empty']:
            idx.remove(empty_idx)

        return idx, f['annotation'], f['filenames'], f['class'], f['empty']
        