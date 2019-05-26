"""
Ad hoc loss functions

Written by Frederic Zhang
Australian National Univeristy

Last updated in May 2019
"""

import torch
import torch.nn as nn

class BCELossForStratifiedBatch(nn.Module):
    """
    Binary cross entropy loss used to deal with multi-label classification
    problems with stratified sampler

    Arguments:
        cfg(CfgNode): configuration class with the following attributes
            cfg.NUM_CLS_PER_BATCH   cfg.POS_GAIN
            cfg.NUM_POS_SAMPLES     cfg.NUM_NEG_SAMPLES
    """
    def __init__(self, cfg):
        super(BCELossForStratifiedBatch, self).__init__()
        self._loss = torch.nn.BCELoss(reduction='none')

        self._stratum_ind = 0
        self._num_strata_batch = cfg.NUM_CLS_PER_BATCH
        self._pos_gain = cfg.POS_GAIN
        self._num_pos = cfg.NUM_POS_SAMPLES
        self._num_neg = cfg.NUM_NEG_SAMPLES

    def _get_stratum_indicator(self, num_cls):
        """
        Get an indicator matrix for the current minibatch

        Classes that stratified sampler has taken sampler from will be marked
        with ones, otherwise zeros
        """
        num_pos_samples = self._num_pos * self._num_strata_batch
        indicator = torch.zeros(num_pos_samples + self._num_neg, num_cls)

        for i in range(self._num_strata_batch):
            indicator[i * self._num_pos: (i + 1) * self._num_pos,
                    (self._stratum_ind + i) % num_cls] = 1

        self._stratum_ind = (self._stratum_ind + self._num_strata_batch) % num_cls

        return indicator

    def forward(self, pred, target):

        loss = self._loss(pred, target)
        
        indicator = self._get_stratum_indicator(target.shape[1]).to(target.device)
        assert torch.sum(indicator * target == indicator) == target.shape[0] * target.shape[1],\
                'Misalignment between indicator matrix and labels'

        mask = indicator * self._pos_gain + (1 - target)

        return torch.sum(loss * mask) / torch.sum(mask)

    def reset(self):
        """Reset the stratum index"""
        self._stratum_ind = 0

class BCEWithLogitsLossForStratifiedBatch(BCELossForStratifiedBatch):
    """
    Binary cross entropy loss coupled with sigmoid function, modified to eal with
    multi-label classification problems with stratified sampler

    Arguments:
        cfg(CfgNode): configuration class with the following attributes
            cfg.NUM_CLS_PER_BATCH   cfg.POS_GAIN
            cfg.NUM_POS_SAMPLES     cfg.NUM_NEG_SAMPLES
    """
    def __init__(self, cfg):
        super(BCELossForStratifiedBatch, self).__init__()
        self._loss = torch.nn.BCEWithLogitsLoss(reduction='none')

        self._stratum_ind = 0
        self._num_strata_batch = cfg.NUM_CLS_PER_BATCH
        self._pos_gain = cfg.POS_GAIN
        self._num_pos = cfg.NUM_POS_SAMPLES
        self._num_neg = cfg.NUM_NEG_SAMPLES

