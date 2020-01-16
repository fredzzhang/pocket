"""
Meters for the purpose of statistics tracking

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import time
import torch
import multiprocessing

from collections import deque
from ..ops import to_tensor

class Meter:
    """
    Base class
    """
    def __init__(self, maxlen=None):
        self._deque = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def __len__(self):
        return len(self._deque)

    def __iter__(self):
        return iter(self._deque)

    def __getitem__(self, i):
        return self._deque[i]

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += str([item for item in self._deque])
        reprstr += ', maxlen='
        reprstr += str(self._maxlen)
        reprstr += ')'
        return reprstr

    def reset(self):
        """Reset the meter"""
        self._deque.clear()

    def append(self, x):
        """Append an element"""
        self._deque.append(x)

    def sum(self):
        """Return the sum of all elements"""
        raise NotImplementedError

    def mean(self):
        """Return the mean"""
        raise NotImplementedError

    def max(self):
        """Return the minimum element"""
        raise NotImplementedError

    def min(self):
        """Return the maximum element"""
        raise NotImplementedError

    @property
    def items(self):
        """Return the content"""
        return [item for item in self._deque]

class NumericalMeter(Meter):
    """
    Meter class with numerals as elements
    """
    VALID_TYPES = [int, float]
    
    def __init__(self, maxlen=None):
        super().__init__(maxlen=maxlen)

    def append(self, x):
        if type(x) in self.VALID_TYPES:
            super().append(x)
        else:
            raise TypeError("Given element \'{}\' is not a numeral".format(x))

    def sum(self):
        return sum(self._deque)

    def mean(self):
        return sum(self._deque) / len(self._deque)

    def max(self):
        return max(self._deque)

    def min(self):
        return min(self._deque)

class HandyTimer(NumericalMeter):
    """
    A timer class that tracks a sequence of time
    """
    def __init__(self, maxlen=None):
        super().__init__(maxlen=maxlen)

    def __enter__(self):
        self._timestamp = time.time()

    def __exit__(self, type, value, traceback):
        self.append(time.time() - self._timestamp)

class AveragePrecisionMeter:
    """
    Meter to compute average precision

    Arguments:
        num_gt(iterable): Number of ground truth instances for each class. When left
            as None, all positives are assumed to have been included in the collected
            results. As a result, full recall is guaranteed when the lowest scoring
            example is accounted for.
        algorithm(str, optional): AP evaluation algorithm
            '11P': 11-point interpolation algorithm prior to voc2010
            'INT': Interpolation algorithm with all points used in voc2010
            'AUC': Precisely as the area under precision-recall curve
        chunksize(int, optional): The approximate size the given iterable will be split
            into for each worker. Use -1 to make the argument adaptive to iterable size
            and number of workers
        output(tensor[N, K], optinoal): Network outputs with N examples and K classes
        labels(tensor[N, K], optinoal): Binary labels

    Usage:
        
    (1) Evalute AP using provided output scores and labels

        >>> # Given output(tensor[N, K]) and labels(tensor[N, K])
        >>> meter = pocket.utils.AveragePrecisionMeter(output=output, labels=labels)
        >>> ap = meter.eval(); map_ = ap.mean()

    (2) Collect results on the fly and evaluate AP

        >>> meter = pocket.utils.AveragePrecisionMeter()
        >>> # Compute output(tensor[N, K]) during forward pass
        >>> meter.append(output, labels)
        >>> ap = meter.eval(); map_ = ap.mean()
        >>> # If you are to start new evaluation and want to reset the meter
        >>> meter.reset()

    """
    def __init__(self, num_gt=None, algorithm="AUC", chunksize=-1, 
            output=None, labels=None):
        self.num_gt = torch.as_tensor(num_gt) \
            if num_gt is not None else None
        self.algorithm = algorithm
        self._chunksize = chunksize
        
        is_none = (output is None, labels is None)
        if is_none == (True, True):
            self._output = torch.Tensor([])
            self._labels = torch.Tensor([])
        elif is_none == (False, False):
            self._output = output.detach().cpu().float()
            self._labels = labels.detach().cpu().float()
        else:
            raise AssertionError("Output and labels should both be given or None")

        self._output_temp = [torch.Tensor([])]
        self._labels_temp = [torch.Tensor([])]

    @staticmethod
    def compute_per_class_ap_as_auc(tuple_):
        """
        Arguments: 
            tuple_[(FloatTensor[N]), (FloatTensor[N])]: precision and recall
        Returns:
            ap(FloatTensor[1])
        """
        prec, rec = tuple_
        ap = 0
        for idx in range(prec.numel()):
            ap +=  prec[idx] * rec[idx] if idx == 0 \
                else 0.5 * (prec[idx] + prec[idx - 1]) * (rec[idx] - rec[idx - 1])
        return ap

    @staticmethod
    def compute_per_class_ap_with_interpolation(tuple_):
        """
        Arguments:
            tuple_[(FloatTensor[N]), (FloatTensor[N])]: precision and recall
        Returns:
            ap(FloatTensor[1])
        """
        prec, rec = tuple_
        ap = 0
        for idx in range(prec.numel()):
            # Precompute max for reuse
            max_ = prec[idx:].max()
            ap +=  max_ * rec[idx] if idx == 0 \
                else 0.5 * (max_ + torch.max(prec[idx - 1], max_)) * (rec[idx] - rec[idx - 1])
        return ap

    @staticmethod
    def compute_per_class_ap_with_11_point_interpolation(tuple_):
        """
        Arguments:
            tuple_[(FloatTensor[N]), (FloatTensor[N])]: precision and recall
        Returns:
            ap(FloatTensor[1])
        """
        prec, rec = tuple_
        ap = 0
        for t in torch.linspace(0, 1, 11):
            inds = torch.nonzero(rec >= t).squeeze()
            if inds.numel():
                ap += (prec[inds].max() / 11)
        return ap

    @classmethod            
    def compute_ap(cls, output, labels, num_gt=None, algorithm='AUC', chunksize=-1):
        """
        Compute AP precisely as the area under the precision-recall curve

        Arguments:
            output(FloatTensor[N, K])
            labels(FloatTensor[N, K])
            num_gt(Tensor[N]): Number of ground truth instances for each class
            algorithm(str): AP evaluation algorithm
            chunksize(int, optional): The approximate size the given iterable will be split
                into for each worker. Use -1 to make the argument adaptive to iterable size
                and number of workers
        Returns:
            ap(FloatTensor[K])
        """
        prec, rec = cls.compute_precision_and_recall(output, labels, 
            num_gt=num_gt)
        ap = torch.zeros(output.shape[1])
        # Use the logic from pool._map_async to compute chunksize
        # https://github.com/python/cpython/blob/master/Lib/multiprocessing/pool.py
        # NOTE: Inappropriate chunksize will cause [Errno 24]Too many open files
        # Make changes with caution
        if chunksize == -1:
            chunksize, extra = divmod(
                    output.shape[1],
                    multiprocessing.cpu_count() * 4)
            if extra:
                chunksize += 1
       
        if algorithm == 'INT':
            algorithm_handle = cls.compute_per_class_ap_with_interpolation
        elif algorithm == '11P':
            algorithm_handle = cls.compute_per_class_ap_with_11_point_interpolation
        elif algorithm == 'AUC':
            algorithm_handle = cls.compute_per_class_ap_as_auc
        else:
            raise ValueError("Unknown algorithm option {}.".format(algorithm))

        with multiprocessing.Pool() as pool:
            for idx, result in enumerate(pool.imap(
                func=algorithm_handle,
                iterable=zip(prec.transpose(0, 1), rec.transpose(0, 1)),
                chunksize=chunksize
            )):
                ap[idx] = result
        
        return ap

    @staticmethod
    def compute_precision_and_recall(output, labels, num_gt=None, eps=1e-8):
        """
        Arguments:
            output(FloatTensor[N, K])
            labels(FloatTensor[N, K])
            num_gt(Tensor[N])
            eps(float): A small constant to avoid division by zero
        Returns:
            prec(FloatTensor[N, K])
            rec(FloatTensor[N, K])
        """
        order = output.argsort(0, descending=True)
        tp = labels[
            order,
            torch.ones_like(order) * torch.arange(output.shape[1])
        ]
        fp = 1 - tp
        tp = tp.cumsum(0)
        fp = fp.cumsum(0)

        prec = tp / (tp + fp)
        # NOTE: The insignificant constant could potentially result in 100%
        # recall being unreachable. Be cautious about its magnitude.
        rec = tp / (labels.sum(0) + eps) if num_gt is None else \
            tp / (num_gt + eps)
        return prec, rec

    def append(self, output, labels):
        """
        Add new results to the meter

        Arguments:
            output(tensor[N, K]): Network output with N examples and K classes
            labels(tensor[N, K]): Binary labels
        """
        if isinstance(output, torch.Tensor) and isinstance(labels, torch.Tensor):
            assert output.shape == labels.shape, \
                "Output scores do not match the dimension of labelss"
            self._output_temp.append(output.detach().cpu().float())
            self._labels_temp.append(labels.detach().cpu().float())
        else:
            raise TypeError("Arguments should both be torch.Tensor")

    def reset(self, keep_old=False):
        """
        Clear saved statistics

        Arguments:
            keep_tracked(bool): If True, clear only the newly collected statistics
                since last evaluation
        """
        if not keep_old:
            self._output = torch.Tensor([])
            self._labels = torch.Tensor([])
        self._output_temp = [torch.Tensor([])]
        self._labels_temp = [torch.Tensor([])]

    def eval(self):
        """
        Evaluate the average precision based on collected statistics

        Returns:
            torch.Tensor[K]: Average precisions for K classes
        """
        self._output = torch.cat([
            self._output,
            torch.cat(self._output_temp, 0)
        ], 0)
        self._labels = torch.cat([
            self._labels,
            torch.cat(self._labels_temp, 0)
        ], 0)
        self.reset(keep_old=True)

        # Sanity check
        if self.num_gt is not None:
            faulty_cls = (self._labels.sum(0) > self.num_gt).nonzero().squeeze(1)
            if len(faulty_cls):
                raise AssertionError("Class {}: ".format(faulty_cls.tolist())+
                    "Number of true positives larger than that of ground truth")
        if len(self._output) and len(self._labels):
            return self.compute_ap(self._output, self._labels, num_gt=self.num_gt,
                algorithm=self.algorithm, chunksize=self._chunksize)
        else:
            print("WARNING: Collected results are empty. "
                "Return zero AP for all class.")
            return torch.zeros(self._output.shape[1])

class DetectionAPMeter:
    """
    A variant of AP meter, where network outputs are assumed to be class-specific.
    Different classes could potentially have different number of samples.

    Required Arguments:
        num_cls(int): Number of target classes
    Optional Arguemnts:
        num_gt(iterable): Number of ground truth instances for each class. When left
            as None, all positives are assumed to have been included in the collected
            results. As a result, full recall is guaranteed when the lowest scoring
            example is accounted for.
        algorithm(str, optional): A choice between '11P' and 'AUC'
            '11P': 11-point interpolation algorithm prior to voc2010
            'INT': Interpolation algorithm with all points used in voc2010
            'AUC': Precisely as the area under precision-recall curve
        chunksize(int, optional): The approximate size the given iterable will be split
            into for each worker. Use -1 to make the argument adaptive to iterable size
            and number of workers
        output(list[tensor], optinoal): A collection of output scores for K classes
        labels(list[tensor], optinoal): Binary labels

    Usage:

    (1) Evalute AP using provided output scores and labels

        >>> # Given output(list[tensor]) and labels(list[tensor])
        >>> meter = pocket.utils.DetectionAPMeter(num_cls, output=output, labels=labels)
        >>> ap = meter.eval(); map_ = ap.mean()

    (2) Collect results on the fly and evaluate AP

        >>> meter = pocket.utils.DetectionAPMeter(num_cls)
        >>> # Get class-specific predictions. The following is an example
        >>> # Assume output(tensor[N, K]) and target(tensor[N]) is given
        >>> pred = output.argmax(1)
        >>> scores = output.max(1)
        >>> meter.append(scores, pred, pred==target)
        >>> ap = meter.eval(); map_ = ap.mean()
        >>> # If you are to start new evaluation and want to reset the meter
        >>> meter.reset()

    """
    def __init__(self, num_cls, num_gt=None, algorithm='AUC', chunksize=-1, 
            output=None, labels=None):
        if num_gt is not None and len(num_gt) != num_cls:
            raise AssertionError("Provided ground truth instances"
                "do not have the same number of classes as specified")

        self.num_cls = num_cls
        self.num_gt = num_gt if num_gt is not None else \
            [None for _ in range(num_cls)]
        self.algorithm = algorithm
        self._chunksize = chunksize

        is_none = (output is None, labels is None)
        if is_none == (True, True):
            self._output = [torch.Tensor([]) for _ in range(num_cls)]
            self._labels = [torch.Tensor([]) for _ in range(num_cls)]
        elif is_none == (False, False):
            assert len(output) == len(labels), \
                "The given output does not have the same number of classes as labels"
            assert len(output) == num_cls, \
                "The number of classes in the given output does not match the argument"
            self._output = to_tensor(output, 
                input_format='list', dtype=torch.float32, device='cpu')
            self._labels = to_tensor(labels,
                input_format='list', dtype=torch.float32, device='cpu')
        else:
            raise AssertionError("Output and labels should both be given or None")

        self._output_temp = [[] for _ in range(num_cls)]
        self._labels_temp = [[] for _ in range(num_cls)]
    
    @classmethod
    def compute_ap(cls, output, labels, num_gt, algorithm='AUC', chunksize=-1):
        """
        Compute AP precisely as the area under the precision-recall curve

        Arguments:
            output(list[FloatTensor])
            labels(list[FloatTensor])
            num_gt(iterable): Number of ground truth instances for each class
            algorithm(str): AP evaluation algorithm
            chunksize(int, optional): The approximate size the given iterable will be split
                into for each worker. Use -1 to make the argument adaptive to iterable size
                and number of workers
        Returns:
            ap(FloatTensor[K])
        """
        ap = torch.zeros(len(output))

        if algorithm == 'INT':
            algorithm_handle = \
                AveragePrecisionMeter.compute_per_class_ap_with_interpolation
        elif algorithm == '11P':
            algorithm_handle = \
                AveragePrecisionMeter.compute_per_class_ap_with_11_point_interpolation
        elif algorithm == 'AUC':
            algorithm_handle = \
                AveragePrecisionMeter.compute_per_class_ap_as_auc
        else:
            raise ValueError("Unknown algorithm option {}.".format(algorithm))

        with multiprocessing.Pool() as pool:
            for idx, result in enumerate(pool.map(
                func=cls.compute_ap_for_each,
                iterable=[(idx, ngt, out, gt, algorithm_handle) 
                    for idx, (ngt, out, gt) in enumerate(zip(num_gt, output, labels))]
            )):
                ap[idx] = result

        return ap

    @classmethod
    def compute_ap_for_each(cls, tuple_):
        idx, num_gt, output, labels, algorithm = tuple_
        # Sanity check
        if num_gt is not None and labels.sum() > num_gt:
            raise AssertionError("Class {}: ".format(idx)+
                "Number of true positives larger than that of ground truth")
        if len(output) and len(labels):
            return algorithm(cls.compute_precision_and_recall(output, labels, num_gt))
        else:
            print("WARNING: Collected results are empty. "
                "Return zero AP for class {}.".format(idx))
            return 0

    @staticmethod
    def compute_precision_and_recall(output, labels, num_gt=None, eps=1e-8):
        """
        Arguments:
            output(FloatTensor[N])
            labels(FloatTensor[N]): Binary labels for each sample
            num_gt(int or float): Number of ground truth instances
            eps(float): A small constant to avoid division by zero
        Returns:
            prec(FloatTensor[N])
            rec(FloatTensor[N])
        """
        order = output.argsort(descending=True)

        tp = labels[order]
        fp = 1 - tp
        tp = tp.cumsum(0)
        fp = fp.cumsum(0)

        prec = tp / (tp + fp)
        rec = tp / (labels.sum() + eps) if num_gt is None else \
            tp / (num_gt + eps)

        return prec, rec

    def append(self, output, prediction, labels):
        """
        Add new results to the meter

        Arguments:
            output(tensor[N]): Output scores for each sample
            prediction(tensor[N]): Predicted classes 0~(K-1)
            labels(tensor[N]): Binary labels for the predicted classes
        """
        if isinstance(output, torch.Tensor) and \
                isinstance(prediction, torch.Tensor) and \
                isinstance(labels, torch.Tensor):
            prediction = prediction.long()
            unique_cls = prediction.unique()
            for cls_idx in unique_cls:
                sample_idx = torch.nonzero(prediction == cls_idx).squeeze(1)
                self._output_temp[cls_idx.item()] += output[sample_idx].tolist()
                self._labels_temp[cls_idx.item()] += labels[sample_idx].tolist()
        else:
            raise TypeError("Arguments should be torch.Tensor")

    def reset(self, keep_old=False):
        """
        Clear saved statistics

        Arguments:
            keep_tracked(bool): If True, clear only the newly collected statistics
                since last evaluation
        """
        num_cls = len(self._output_temp)
        if not keep_old:
            self._output = [torch.Tensor([]) for _ in range(num_cls)]
            self._labels = [torch.Tensor([]) for _ in range(num_cls)]
        self._output_temp = [[] for _ in range(num_cls)]
        self._labels_temp = [[] for _ in range(num_cls)]

    def eval(self):
        """
        Evaluate the average precision based on collected statistics

        Returns:
            torch.Tensor[K]: Average precisions for K classes
        """
        self._output = [torch.cat([
            out1, torch.as_tensor(out2, dtype=torch.float32)
        ]) for out1, out2 in zip(self._output, self._output_temp)]
        self._labels = [torch.cat([
            tar1, torch.as_tensor(tar2, dtype=torch.float32)
        ]) for tar1, tar2 in zip(self._labels, self._labels_temp)]
        self.reset(keep_old=True)

        return self.compute_ap(self._output, self._labels, self.num_gt,
            algorithm=self.algorithm, chunksize=self._chunksize)
