"""
Default configurations based on YACS 

Written by Frederic Zhang
Australian National University

Last updated in Apr. 2019
"""

import os
from yacs.config import CfgNode as CN

_C = CN()


"""Configurations concerning datasets/dataloader"""

_C.DATA = CN()

# Name of the dataset
_C.DATA.NAME = ''
# Root directory of the dataset
_C.DATA.ROOT = ''
# Metadata directory
_C.DATA.METADIR ''
# Number of classes
_C.DATA.NUM_CLASSES = 1
# Minimum IoU to identify same bounding boxes
_C.DATA.MIN_IOU = .5
# Interval between two progress updates when constructing dataset
_C.DATA.INTVL = 500

_C.DATA.TRAIN = CN()
_C.DATA.TEST = CN()

# Directory of dataset images
_C.DATA.TRAIN.IMAGES = ''
_C.DATA.TEST.IMAGES = ''
# Path of pickle files for saving and loading image database
_C.DATA.TRAIN.IMDB = ''
_C.DATA.TEST.IMDB = ''

# Directory of G.T. box pair files
_C.DATA.TRAIN.GTDIR = ''
_C.DATA.TEST.GTDIR = ''
# Path of pickle files for saving and loading ground truth database
_C.DATA.TRAIN.GTDB = ''
_C.DATA.TEST.GTDB = ''

# Directory of bounding box detections
_C.DATA.TRAIN.DETDIR = ''
_C.DATA.TEST.DETDIR = ''
# Path of pickle files for saving and loading detection database
_C.DATA.TRAIN.DETDB = ''
_C.DATA.TEST.DETDB = ''


"""Configurations concerning model structure"""

_C.MODEL = CN()



"""Configurations concerning experiments"""

_C.EXPT = CN()


def load_default_cfgs():
    return _C.clone()
