"""
HICODet dataset under PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json

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
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version.
    """
    def __init__(self, root, annoFile, transform=None, target_transform=None, transforms=None):
        super(HICODet, self).__init__(root, transform, target_transform, transforms)
        with open(annoFile, 'r') as f:
            anno = json.load(f)

        self.num_object_cls = 80
        self.num_interation_cls = 600
        self.num_action_cls = 117
        self._annoFile = annoFile

        # Load annotations
        self.load_annotation_and_metadata(anno)

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
        return self._transforms(
            self.load_image(os.path.join(self._root, self._filenames[intra_idx])), 
            self._anno[intra_idx]
            )

    def __repr__(self):
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=\"' + repr(self._root)
        reprstr += '\", annoFile=\"'
        reprstr += repr(self._annoFile)
        reprstr += '\")'
        # Ignore the optional arguments
        return reprstr


    def __str__(self):
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._root)
        return reprstr

    @property
    def class_corr(self):
        """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
        return self._class_corr.copy()

    @property
    def object_to_interaction(self):
        """
        The interaction classes that involve each object type
        
        Returns:
            list[list]
        """
        return self._obj_to_int.copy()

    @property
    def anno_interaction(self):
        """
        Number of annotated box pairs for each interaction class

        Returns:
            list[600]
        """
        return self._num_anno.copy()

    @property
    def anno_object(self):
        """
        Number of annotated box pairs for each object class

        Returns:
            list[80]
        """
        num_anno = [0 for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            num_anno[corr[1]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def anno_action(self):
        """
        Number of annotated box pairs for each action class

        Returns:
            list[117]
        """
        num_anno = [0 for _ in range(self.num_action_cls)]
        for corr in self._class_corr:
            num_anno[corr[2]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def objects(self):
        """
        Object names 

        Returns:
            list[str]
        """
        return self._objects

    @property
    def verbs(self):
        """
        Verbs (action names)

        Returns:
            list[str]
        """
        return self._verbs

    @property
    def interactions(self):
        """
        Combination of verbs and objects

        Returns:
            list[str]
        """
        return [self._verbs[j] + ' ' + self.objects[i] 
            for _, i, j in self._class_corr]


    def filename(self, idx):
        """Return the image file name"""
        return self._filenames[self._idx[idx]]

    def load_annotation_and_metadata(self, f):
        """
        Arguments:
            f(dict): Dictionary loaded from {annoFile}.json
        """
        idx = list(range(len(f['filenames'])))
        for empty_idx in f['empty']:
            idx.remove(empty_idx)
        
        obj_to_int = [[] for _ in range(self.num_object_cls)]
        for corr in f['correspondence']:
            obj_to_int[corr[1]].append(corr[0])

        num_anno = [0 for _ in range(self.num_interation_cls)]
        for anno in f['annotation']:
            for hoi in anno['hoi']:
                num_anno[hoi] += 1

        self._idx = idx
        self._obj_to_int = obj_to_int
        self._num_anno = num_anno

        self._anno = f['annotation']
        self._filenames = f['filenames']
        self._class_corr = f['correspondence']
        self._empty_idx = f['empty']
        self._objects = f['objects']
        self._verbs = f['verbs']
