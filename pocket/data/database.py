"""
Dataset base classes

Written by Frederic Zhang
Australian National University

Last updated in Apr. 2019
"""

import os
import cv2
import pickle
import numpy as np

class DataDict(dict):
    """
    Data dictionary class

    FOR BYTE-LIKE DATA ONLY
    """
    def __init__(self, dictargs={}):
        """Constructor method"""
        super(DataDict, self).__init__(**dictargs)

    def __getattr__(self, name):
        """Get attribute"""
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        """Set attribute"""
        self[name] = value

    def save(self, path):
        """Save the dictionary into a pickle file"""
        with open(path, 'wb') as f:
            pickle.dump(self.copy(), f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """Load from a dict or DataDict class"""
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        for name in data_dict:
            self[name] = data_dict[name]

    def is_empty(self):
        return not bool(len(self))

class Imdb:
    """
    Base class for image database

    Arguments:
        subset(str): subset name, typically a choice between
            'TRAIN' and 'TEST'
        cfg(CfgNode): configuration class with the following attributes
            cfg.NAME
            cfg.TRAIN.IMAGES        cfg.TEST.IMAGES
            cfg.TRAIN.IMDB          cfg.TEST.IMDB
    """
    def __init__(self, subset, cfg):
        """Constructor method"""
        self._subset = subset
        self._imdb_name = cfg.NAME
        self._image_paths = self._construct_image_paths(cfg[subset].IMAGES)
        self._flip = np.zeros(len(self._image_paths), dtype=np.bool)
        self._num_images = len(self._image_paths)
        self._num_classes = None
        self._imdb = DataDict()
        self._cfg = cfg

    def __len__(self):
        """Return the number of images by default"""
        return self._num_images

    def __getitem__(self, i):
        """Return the image as ndarray by default"""
        return self.fetch_image(i)
    
    @property
    def name(self):
        """Return database name and subset name"""
        return '{} {}'.format(self._imdb_name, self._subset)
    
    @property
    def num_images(self):
        """Return the total number of images in the database"""
        return self._num_images

    @property
    def num_classes(self):
        """Return the number of classes of images"""
        return self._num_classes
    
    @property
    def imdb(self):
        """Return the image database"""
        if self._imdb.is_empty():
            self._load_imdb()
        return self._imdb

    @staticmethod
    def _construct_image_paths(src_dir):
        """Get the list for the image paths"""
        im_paths =  [os.path.join(src_dir, f) for f in os.listdir(src_dir) \
                if f.endswith('.jpg') or f.endswith('png')]
        im_paths.sort()
        return im_paths

    def _construct_imdb(self):
        """Construct image database"""
        raise NotImplementedError

    def _load_imdb(self):
        """Load or construct the image database"""
        if os.path.exists(self._cfg[self._subset].IMDB):
            print('\nLOADING IMDB FROM CACHE...\n')
            self._imdb.load(self._cfg[self._subset].IMDB)
            print('\nDONE\n')
        else:
            print('\nCONSTRUCTING IMDB...\n')
            self._construct_imdb()
            self._imdb.save(self._cfg[self._subset].IMDB)
            print('\nDONE\n')

    def _flip_images(self):
        """Flip the images for data augmentation"""
        self._image_paths = self._image_paths * 2
        self._num_images *= 2
        self._flip = np.concatenate([self._flip,
            np.ones(self._num_images, dtype=np.bool)])

    def image_path(self, i):
        """Return the path of an image given the index"""
        return self._image_paths[i]

    def fetch_image(self, i):
        """Return an image in ndarray given the index"""
        im =  cv2.imread(self._image_paths[i])
        if self._flip[i]:   return im[:, ::-1, :]
        else:   return im

class Detdb(Imdb):
    """
    Base class of bounding-box-based detection database

    By default, the ground truth and detection bounding boxes are
    assumed to have been written in .txt files, with the directory
    paths specified in the argument list. For different file formats,
    override the following methods:

        self._construct_src_paths()
        self._load_from_src_files()

    Arguments:
        subset(str): subset name, typically a choice between
            'TRAIN' and 'TEST''
        cfg(CfgNode): configuration class with the following attributes
            cfg.NAME
            cfg.INTVL
            cfg.NUM_CLASSES
            cfg.TRAIN.IMAGES        cfg.TEST.IMAGES
            cfg.TRAIN.IMDB          cfg.TEST.IMDB
            cfg.TRAIN.GTDIR         cfg.TEST.GTDIR
            cfg.TRAIN.GTDB          cfg.TEST.GTDB
            cfg.TRAIN.DETDIR        cfg.TEST.DETDIR
            cfg.TRAIN.DETDB         cfg.TEST.DETDB
    """
    def __init__(self, subset, cfg):
        """Constructor method"""
        super(Detdb, self).__init__(subset, cfg)
        self._num_classes = cfg.NUM_CLASSES
        self._gtdb = DataDict()
        self._detdb = DataDict()
        self._fg_obj = None

    @property
    def gtdb(self):
        """Return the ground truth database"""
        if self._gtdb.is_empty():
            self._load_ground_truth()
        return self._gtdb

    @property
    def detdb(self):
        """Return the detection database"""
        if self._detdb.is_empty():
            self._load_detdb()
        return self._detdb

    @staticmethod
    def _construct_src_paths(src_dir):
        """Get the list of all .txt files"""
        txt_paths = [os.path.join(src_dir, f) for f in os.listdir(src_dir) \
                if f.endswith('.txt')]
        txt_paths.sort()
        return txt_paths
    
    def _load_from_src_files(self, txt_files):
        """Load data from .txt files as ndarray"""
        db = np.empty(self._num_images, dtype=object)
        for i, f in enumerate(txt_files):
            db[i] = np.loadtxt(f, ndmin=2)
            if i % self._cfg.INTVL == 0:
                print('Progress: {:>6d}/{:>6d}'.format(i + 1, len(txt_files)))
        return db

    def _load_ground_truth(self):
        """Load or construct ground truth database"""
        if os.path.exists(self._cfg[self._subset].GTDB):
            print('\nLOADING G.T. DATABASE FROM CACHE...\n')
            self._gtdb.load(self._cfg[self._subset].GTDB)
            print('\nDONE\n')
        else:
            gt_files = self._construct_src_paths(self._cfg[self._subset].GTDIR)
            print('\nLOADING G.T. DATA...\n')
            gt_data = self._load_from_src_files(gt_files)
            print('\nCONSTRUCTING G.T. DATABASE...\n')
            self._construct_gtdb(gt_data)
            self._gtdb.save(self._cfg[self._subset].GTDB)
            print('\nDONE\n')

    def _load_detdb(self):
        """Load or construct detection database"""
        self._load_ground_truth()
        if os.path.exists(self._cfg[self._subset].DETDB):
            print('\nLOADING DETECTION DATABASE FROM CACHE...\n')
            self._detdb.load(self._cfg[self._subset].DETDB)
            print('\nDONE\n')
        else:
            det_files = self._construct_src_paths(self._cfg[self._subset].DETDIR)
            print('\nLOADING DETECTION DATA...\n')
            det_data = self._load_from_src_files(det_files)
            print('\nCONSTRUCTING DETECTION DATABASE\n')
            self._construct_detdb(det_data)
            self._detdb.save(self._cfg[self._subset].DETDB)
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

    def _flip_all(self):
        """Flip the images and bounding boxes"""
        raise NotImplementedError
