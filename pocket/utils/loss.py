"""
Loss functions and related utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineWeightAdjustment:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self._register = torch.zeros(num_classes)

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += 'num_classes='
        reprstr += repr(self.num_classes)
        reprstr += ')'
        return reprstr

    def compute_weights(self, class_idx, labels, num_iter=10):
        """
        Compute weight for each logit

        Arguments:
            class_idx(LongTensor[N])
            labels(FloatTensor[N]): Binary labels
            num_iter(int): Number of iterations to update the weights
        Returns:
            weights(FloatTensor[N]): Weights to be applied to each logit
        """
        device = labels.device
        class_idx = class_idx.clone().cpu()
        labels = labels.clone().cpu()

        weights = torch.ones_like(labels)
        modify_idx = []; parameters = []
        for i in torch.nonzero(self._register).squeeze(1):
            bias = self._register[i]

            logits_idx = torch.nonzero(class_idx == i).squeeze(1)
            # Skip when there are not logits from the current class
            if not len(logits_idx):
                continue
            if bias > 0:
                keep_idx = logits_idx[torch.nonzero(labels[logits_idx] == 0)].squeeze(1)
            else:
                keep_idx = logits_idx[torch.nonzero(labels[logits_idx])].squeeze(1)
            # Skip when there are not logits that contribute positively to the balance
            if not len(keep_idx):
                continue
            p = len(keep_idx); n = len(logits_idx) - p
            # Log indices and parameters to compute weights
            modify_idx.append(keep_idx)
            parameters.append(torch.cat([
                p * torch.ones(p, 1), n * torch.ones(p, 1), bias.abs().repeat(p, 1)
            ], 1))

        if len(modify_idx):
            modify_idx = torch.cat(modify_idx)
            p, n, bias = torch.cat(parameters).unbind(1)
            # Iteratively update the weights
            for _ in range(num_iter):
                sum_ = weights.sum().item()
                weights[modify_idx] = (n + sum_ * bias) / p
        # Normalise the weights
        weights /= weights.sum()

        return weights.to(device)

    def update_register(self, class_idx, labels, weights):
        """
        Update register

        Arguments:
            class_idx(LongTensor[N])
            labels(FloatTensor[N]): Binary labels
            weights(FloatTensor[N]): Weights applied to each logit
        """
        class_idx = class_idx.clone().cpu()
        weights = weights.clone().cpu()

        labels = labels.clone().cpu()
        labels = labels * 2 - 1

        total_bias = labels * weights

        unique_class = class_idx.unique()
        for idx in unique_class:
            logits_idx = torch.nonzero(class_idx == idx).squeeze(1)
            self._register[idx] += (total_bias[logits_idx]).sum()

class PairwiseSoftMarginLoss(nn.Module):
    r"""
    For each class, take every positive/negative pair and compute the soft margin loss based
        one the formula log(1 + exp(-xy))

    Arguments:
        reduction(str, optional): Specifies the reduction to apply to the output. Choose
            between 'none', 'mean' and 'sum'
        sampling_ratio(float, optional): Ratio of positive-negative pairs to sample

    Example:

        >>> import torch
        >>> from pocket.utils import PairwiseSoftMarginLoss
        >>> # Generate predictions and targets
        >>> pred = torch.tensor([[0.0706, 0.8677],[0.6481, 0.7287], [0.2627, 0.0302]])
        >>> target = torch.tensor([[0., 1.], [1., 0.], [0., 1.]])
        >>> # Initialize loss function
        >>> criterion = PairwiseSoftMarginLoss()
        >>> # Compute loss
        >>> criterion(pred, target)
        tensor(0.4185)
    """
    def __init__(self, reduction='mean', sampling_ratio=1):
        assert reduction in ['none', 'mean', 'sum'],\
                'Unsupported reduction mode {}. Use \'none\', \'mean\' or \'sum\''.format(reduction)
        super(PairwiseSoftMarginLoss, self).__init__()
        self._reduction = reduction
        self._sampling_ratio = sampling_ratio

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += repr(self._reduction)
        reprstr += ', '
        reprstr += repr(self._sampling_ratio)
        reprstr += ')'
        return reprstr

    def forward(self, pred, target):
        """
        Arguments:
            pred(Tensor[N, C]): Predicted scores of N samples for C classes
            target(Tensor[N, C]): Binary labels
        """
        # construct positive/negative pairs in cpu due to high memory consumption
        target_cpu = target.clone().cpu()

        # find the indices of all positive and negatie logits
        all_p = target.nonzero()
        all_n = (1 - target).nonzero()

        # pair up positive and negative logits regardless of class
        all_p_x, all_n_x = torch.meshgrid(all_p[:, 0], all_n[:, 0])
        all_p_y, all_n_y = torch.meshgrid(all_p[:, 1], all_n[:, 1])

        # only keep pairs from the same class
        mask = torch.eq(all_p_y, all_n_y)
        p_x = all_p_x[mask].to(pred.device)
        p_y = all_p_y[mask].to(pred.device)
        n_x = all_n_x[mask].to(pred.device)
        n_y = all_n_y[mask].to(pred.device)

        # randomly sample a proportion of pairs
        if self._sampling_ratio != 1:
            sample_inds = torch.randperm(p_x.numel())[:int(p_x.numel() * self._sampling_ratio)]
            p_x = p_x[sample_inds]
            p_y = p_x[sample_inds]
            n_x = p_x[sample_inds]
            n_y = p_x[sample_inds]

        loss = F.soft_margin_loss(
                pred[p_x, p_y] - pred[n_x, n_y],
                torch.ones_like(p_x, dtype=torch.float32, device=pred.device),
                reduction='none'
                )

        if self._reduction == 'none':
            return loss
        elif self._reduction == 'mean':
            return torch.mean(loss)
        elif self._reduction == 'sum':
            return torch.sum(loss)
        else:
            raise NotImplementedError('Unsupported reduction mode {}'.format(self._reduction))

class PSMLWithBCE(PairwiseSoftMarginLoss):

    def __init__(self, sampling_ratio=1):
        super(PSMLWithBCE, self).__init__('mean', sampling_ratio)

    def forward(self, pred, target):
        return super(PSMLWithBCE, self).forward(pred, target) \
                + F.binary_cross_entropy_with_logits(
                        pred,
                        target,
                        reduction='mean')
        
class PairwiseMarginRankingLoss(nn.Module):
    r"""
    For each class, take every positive/negative pair and compute the max-margin loss.

    Arguments:
        margin(float, optional): Should be a number from -1 to 1, 0 to 0.5 is suggested. The
            default is 0
        reduction(str, optional): Specifies the reduction to apply to the output. Choose
            between 'none', 'mean' and 'sum'
        remove_easy(bool, optional): Remove easy pairs with zero loss
        sampling_ratio(float, optional): Ratio of positive-negative pairs to sample

    Example:

        >>> import torch
        >>> from pocket.utils import PairwiseMarginRankingLoss
        >>> # Generate predictions and targets
        >>> pred = torch.tensor([[0.0706, 0.8677],[0.6481, 0.7287], [0.2627, 0.0302]])
        >>> target = torch.tensor([[0., 1.], [1., 0.], [0., 1.]])
        >>> # Initialize loss function
        >>> criterion = PairwiseMarginRankingLoss(0.5)
        >>> # Compute loss
        >>> criterion(pred, target)
        tensor(0.4185)
    """
    def __init__(self, margin=0.0, reduction='mean', remove_easy=False, sampling_ratio=1):
        assert reduction in ['none', 'mean', 'sum'],\
                'Unsupported reduction mode {}. Use \'none\', \'mean\' or \'sum\''.format(reduction)
        super(PairwiseMarginRankingLoss, self).__init__()
        self._margin = margin
        self._reduction = reduction
        self._remove_easy = remove_easy
        self._sampling_ratio = sampling_ratio

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += repr(self._margin)
        reprstr += ', '
        reprstr += repr(self._reduction)
        reprstr += ', '
        reprstr += repr(self._remove_easy)
        reprstr += ', '
        reprstr += repr(self._sampling_ratio)
        reprstr += ')'
        return reprstr

    def forward(self, pred, target):
        """
        Arguments:
            pred(Tensor[N, C]): Predicted scores of N samples for C classes
            target(Tensor[N, C]): Binary labels
        """
        # normalize the scores using sigmoid
        #pred = torch.sigmoid(pred)

        # construct positive/negative pairs in cpu due to high memory consumption
        target_cpu = target.clone().cpu()

        # find the indices of all positive and negatie logits
        all_p = target_cpu.nonzero()
        all_n = (1 - target_cpu).nonzero()

        # pair up positive and negative logits regardless of class
        all_p_x, all_n_x = torch.meshgrid(all_p[:, 0], all_n[:, 0])
        all_p_y, all_n_y = torch.meshgrid(all_p[:, 1], all_n[:, 1])

        # only keep pairs from the same class
        mask = torch.eq(all_p_y, all_n_y)
        p_x = all_p_x[mask].to(pred.device)
        p_y = all_p_y[mask].to(pred.device)
        n_x = all_n_x[mask].to(pred.device)
        n_y = all_n_y[mask].to(pred.device)

        # randomly sample a proportion of pairs
        if self._sampling_ratio != 1:
            sample_inds = torch.randperm(p_x.numel())[:int(p_x.numel() * self._sampling_ratio)]
            p_x = p_x[sample_inds]
            p_y = p_x[sample_inds]
            n_x = p_x[sample_inds]
            n_y = p_x[sample_inds]

        loss = F.margin_ranking_loss(
                pred[p_x, p_y],
                pred[n_x, n_y],
                torch.ones_like(p_x, dtype=torch.float32, device=pred.device),
                margin=self._margin,
                reduction='none'
                )

        # remove easy pairs
        if self._remove_easy:
            loss = loss[torch.nonzero(loss)[:, 0]]

        if self._reduction == 'none':
            return loss
        elif self._reduction == 'mean':
            return torch.mean(loss)
        elif self._reduction == 'sum':
            return torch.sum(loss)
        else:
            raise NotImplementedError('Unsupported reduction mode {}'.format(self._reduction))

# NOTE: Deprecated
class PMRLWithOHEM(PairwiseMarginRankingLoss):
    """
    Pairwise margin ranking loss with online hard example mining

    Arguments:
        margin(float, optional): Should be a number from -1 to 1, 0 to 0.5 is suggested. The
            default is 0
        reduction(str, optional): Specifies the reduction to apply to the output. Choose
            between 'none', 'mean' and 'sum'
        keep(int, optional): Number of hard examples kept for back propagation
        positive_ratio(float, optional): A rough ratio of per-class positives used for memory
            pre-allocation. Use 1 if unsure
    """
    def __init__(self, margin=0.0, reduction='mean', keep=64, positive_ratio=0.2):
        super(PMRLWithOHEM, self).__init__(margin, 'none', positive_ratio)
        self._keep = keep
        self._child_reduction = reduction

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += repr(self._margin)
        reprstr += ', '
        reprstr += repr(self._child_reduction)
        reprstr += ', '
        reprstr += repr(self._keep)
        reprstr += ', '
        reprstr += repr(self._poitive_ratio)
        reprstr += ')'
        return reprstr

    def forward(self, pred, target):
        loss = super(PMRLWithOHEM, self).forward(pred, target)
        hard_inds = torch.argsort(loss)[-self._keep:]
        if self._child_reduction == 'mean':
            return torch.mean(loss[hard_inds])
        elif self._child_reduction == 'sum':
            return torch.sum(loss[hard_inds])
        elif self._child_reduction == 'none':
            return loss[hard_inds]
        else:
            raise NotImplementedError('Unsupported reduction mode {}'.format(self._reduction))


class BCELossForStratifiedBatch(nn.Module):
    """
    Binary cross entropy loss modified to deal with multi-label classification
    problems with stratified sampler. Co-occurring classes that are not the sampled
    class will be masked out

    Arguments:
        sampler_cfgs(CfgNode): Configuration of the stratified sampler with the
            following attributes
                NUM_CLS_PER_BATCH
                NUM_SAMPLES_PER_CLS
                NUM_NEG_SAMPLES
                NUM_BATCHES_PER_EPOCH
        pos_gain(float, optional): Gain applied on loss digits of positive samples
    """
    def __init__(self, sampler_cfgs, pos_gain=1):
        super(BCELossForStratifiedBatch, self).__init__()
        self._loss = torch.nn.BCELoss(reduction='none')

        self._stratum_ind = 0
        self._batch_ind = 0
        self._cfgs = sampler_cfgs
        self._pos_gain = pos_gain

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += repr(self._cfgs)
        reprstr += ', '
        reprstr += repr(self._pos_gain)
        reprstr += ')'
        return reprstr

    def _get_stratum_indicator(self, num_cls):
        """
        Get an indicator matrix for the current minibatch

        Classes that stratified sampler has taken sampler from will be marked
        with ones, otherwise zeros
        """
        self._batch_ind += 1

        num_pos_per_batch = self._cfgs.NUM_CLS_PER_BATCH * self._cfgs.NUM_SAMPLES_PER_CLS
        indicator = torch.zeros(num_pos_per_batch + self._cfgs.NUM_NEG_SAMPLES, num_cls)

        for i in range(self._cfgs.NUM_CLS_PER_BATCH):
            indicator[i * self._cfgs.NUM_SAMPLES_PER_CLS: (i + 1) * self._cfgs.NUM_SAMPLES_PER_CLS,
                    (self._stratum_ind + i) % num_cls] = 1
        self._stratum_ind = (self._stratum_ind + self._cfgs.NUM_CLS_PER_BATCH) % num_cls

        if self._batch_ind == self._cfgs.NUM_BATCHES_PER_EPOCH:
            self._stratum_ind = 0
            self._batch_ind = 0

        return indicator

    def forward(self, pred, target):

        loss = self._loss(pred, target)
        
        indicator = self._get_stratum_indicator(target.shape[1]).to(target.device)
        assert torch.sum(indicator * target == indicator) == target.shape[0] * target.shape[1],\
                'Misalignment between indicator matrix and labels at {}'.\
                format(torch.nonzero(indicator * target != indicator))

        mask = indicator * self._pos_gain + (1 - target)

        return torch.sum(loss * mask) / torch.sum(mask)


class BCEWithLogitsLossForStratifiedBatch(BCELossForStratifiedBatch):
    """
    Binary cross entropy loss coupled with sigmoid function, modified to eal with
    multi-label classification problems with stratified sampler

    Arguments:
        sampler_cfgs(CfgNode): Configuration of the stratified sampler with the
            following attributes
                NUM_CLS_PER_BATCH
                NUM_SAMPLES_PER_CLS
                NUM_NEG_SAMPLES
                NUM_BATCHES_PER_EPOCH
        pos_gain(float, optional): Gain applied on loss digits of positive samples
    """
    def __init__(self, sampler_cfgs, pos_gain=1):
        super(BCELossForStratifiedBatch, self).__init__()
        self._loss = torch.nn.BCEWithLogitsLoss(reduction='none')

        self._stratum_ind = 0
        self._batch_ind = 0
        self._cfgs = sampler_cfgs
        self._pos_gain = pos_gain

