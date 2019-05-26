"""
Common I/O utilities

Written by Frederic Zhang
Australian National University

Last updated in Mar. 2019
"""

import yaml
import pickle
import datetime

def load_cfg_from(file_path):
    with open(file_path, 'r') as f:
        cfg = yaml.load(f)
    return cfg

def load_pkl_from(file_path, mode='rb'):
    with open(file_path, mode) as f:
        pkl = pickle.load(f)
    return pkl

class Log:
    """
    Logger

    Arguments:
        path(str): path of the file to write to
        mode(str): file editing mode
    """
    def __init__(self, path, mode='wb'):
        self._path = path
        self._mode = mode

    @property
    def path(self):
        """Return the path of the log"""
        return self._path

    @property
    def mode(self):
        """Return the current file editing  mode"""
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        """Alter the file editing mode"""
        self._mode = new_mode

    def write(self, descpt):
        """Write to file"""
        with open(self._path, self._mode) as f:
            f.write(descpt)

    def time(self):
        """Print current time to log"""
        with open(self._path, self._mode) as f:
            f.write('\n' + str(datetime.datetime.now()))


class TrainLog(Log):
    """
    Logger for training stats

    Arguments:
        path(str): path of the file to write to
        mode(str): file editing mode
    """
    def __init__(self, path, mode='a'):
        super(TrainLog, self).__init__(path, mode)

    def log(self, step, train_loss, val_loss=None):
        """Write training log"""
        self.time()
        if val_loss is None:
            self.write('\nStep: {}, training loss: {}\n'.format(step, train_loss))
        else:
            self.write('\nStep: {}, training loss: {}, validation loss: {}\n'.format(step, train_loss, val_loss))
