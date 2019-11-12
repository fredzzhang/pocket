"""
Dataset base classes

Written by Frederic Zhang
Australian National University

Last updated in Aug. 2019
"""

import os
import cv2
import pickle
import warnings
import numpy as np

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

class ImageDataset:
    r"""
    Base class for image dataset

    Arguments:
        subset(str): Subset name
        cfg(CfgNode): Configuration class with the following attributes
            cfg.DATASET.NAME
            cfg.DATASET.{subset}.IMAGE_DIR
    """
    def __init__(self, subset, cfg):
        """Constructor method"""
        self._subset = str(subset)
        self._cfg = cfg

        self._image_paths = self.construct_image_paths(cfg.DATASET[subset].IMAGE_DIR)
        self._num_images = len(self._image_paths)

        self._num_classes = None
        self._flip = np.zeros(len(self._image_paths), dtype=np.bool)

    def __len__(self):
        """Return the number of images"""
        return self._num_images

    def __getitem__(self, i):
        """Return an image given the index"""
        return self.fetch_image(i)
    
    def __repr__(self):
        """Return the string representation"""
        reprstr = self.__class__.__name__ + '(' + repr(self._subset) + ', '
        reprstr += repr(self._cfg)
        reprstr += ')'
        return reprstr

    def __str__(self):
        """Return database name and subset name"""
        return '{}-{}'.format(self._cfg.DATASET.NAME, self._subset)
    
    @property
    def num_images(self):
        """Number of images"""
        return self._num_images

    @property
    def num_classes(self):
        """Number of classes of in the image dataset"""
        if self._num_classes is None:
            raise NotImplementedError('Number of classes is not specified')
        else:
            return self._num_classes
    
    @staticmethod
    def construct_image_paths(src_dir):
        """Get the list for the image paths"""
        im_paths =  [os.path.join(src_dir, f) for f in os.listdir(src_dir) \
                if f.endswith('.jpg') or f.endswith('.png')]
        im_paths.sort()
        return im_paths

    def flip_images(self):
        """Flip the images for data augmentation"""
        self._image_paths *= 2
        self._num_images *= 2
        self._flip = np.concatenate([self._flip,
            np.ones(self._num_images, dtype=np.bool)])

    def image_path(self, i):
        """Return the path of an image given the index"""
        return self._image_paths[i]

    def fetch_image(self, i):
        """Return an image [H, W, C] in numpy.ndarray given the index"""
        im =  cv2.imread(self._image_paths[i])
        if self._flip[i]:   return im[:, ::-1, :]
        else:   return im

class DetectionDataset(ImageDataset):
    r"""
    Base class for bounding-box detection dataset

    By default, the ground truth and detection bounding boxes are
    assumed to have been written in .txt files, with the directory
    paths specified in the argument list. 

    Arguments:
        subset(str): Subset name
        cfg(CfgNode): Configuration class with the following attributes
            cfg.DATASET.NAME
            cfg.DATASET.SOURCE_FILE_EXTENSION
            cfg.DATASET.PRINT_INTERVAL
            cfg.DATASET.{subset}.IMAGE_DIR
            cfg.DATASET.{subset}.GT_DIR
            cfg.DATASET.{subset}.GT_CACHE_PATH
            cfg.DATASET.{subset}.DETECTION_DIR
            cfg.DATASET.{subset}.DETECTION_CACHE_PATH
    """
    def __init__(self, subset, cfg):
        """Constructor method"""
        super(DetectionDataset, self).__init__(subset, cfg)
        self._gtdb = DataDict()
        self._detdb = DataDict()
        self._fg_obj = None

    @property
    def gtdb(self):
        """The ground truth database"""
        if self._gtdb.is_empty():
            self._load_ground_truth()
        return self._gtdb

    @property
    def detdb(self):
        """The detection database"""
        if self._detdb.is_empty():
            self._load_detection()
        return self._detdb

    @staticmethod
    def construct_src_paths(src_dir, ext='.txt'):
        """Get the list of all files with a specified extension"""
        txt_paths = [os.path.join(src_dir, f) for f in os.listdir(src_dir) \
                if f.endswith(ext)]
        txt_paths.sort()
        return txt_paths
    
    def _load_from_src_files(self, txt_files):
        """Load data from .txt files into a list"""
        db = []
        for i, f in enumerate(txt_files):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                db.append(np.loadtxt(f, ndmin=2))
            if i % self._cfg.DATASET.PRINT_INTERVAL == 0:
                print('Progress: {:>6d}/{:>6d}'.format(i + 1, len(txt_files)))
        return db

    def _load_ground_truth(self):
        """Load or construct ground truth database"""
        if os.path.exists(self._cfg.DATASET[self._subset].GT_CACHE_PATH):
            print('\nLOADING G.T. DATABASE FROM CACHE...\n')
            self._gtdb.load(self._cfg.DATASET[self._subset].GT_CACHE_PATH)
            print('\nDONE\n')
        else:
            gt_files = self.construct_src_paths(
                    self._cfg.DATASET[self._subset].GT_DIR,
                    self._cfg.DATASET.SOURCE_FILE_EXTENSION)
            print('\nLOADING G.T. DATA...\n')
            gt_data = self._load_from_src_files(gt_files)
            print('\nCONSTRUCTING G.T. DATABASE...\n')
            self._construct_gtdb(gt_data)
            self._gtdb.save(self._cfg.DATASET[self._subset].GT_CACHE_PATH)
            print('\nDONE\n')

    def _load_detection(self):
        """Load or construct detection database"""
        if self._gtdb.is_empty():
            self._load_ground_truth()
        if os.path.exists(self._cfg.DATASET[self._subset].DETECTION_CACHE_PATH):
            print('\nLOADING DETECTION DATABASE FROM CACHE...\n')
            self._detdb.load(self._cfg.DATASET[self._subset].DETECTION_CACHE_PATH)
            print('\nDONE\n')
        else:
            det_files = self.construct_src_paths(
                    self._cfg.DATASET[self._subset].DETECTION_DIR,
                    self._cfg.DATASET.SOURCE_FILE_EXTENSION)
            print('\nLOADING DETECTION DATA...\n')
            det_data = self._load_from_src_files(det_files)
            print('\nCONSTRUCTING DETECTION DATABASE\n')
            self._construct_detdb(det_data)
            self._detdb.save(self._cfg.DATASET[self._subset].DETECTION_CACHE_PATH)
            print('\nDONE\n')

    def fg_obj(self, i):
        """Return the foreground object indices of an image"""
        if self._fg_obj is None:
            raise NotImplementedError
        return self._fg_obj[i]

    def _construct_gtdb(self, gt_data):
        """Construct ground truth database"""
        raise NotImplementedError

    def _construct_detdb(self, det_data):
        """Construct detection database"""
        raise NotImplementedError
