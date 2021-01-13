"""
Meters for the purpose of statistics tracking

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import time
import torch
import multiprocessing

from torch import Tensor
from collections import deque
from typing import Optional, Iterable, Any, List, Union, Tuple
from ..ops import to_tensor

__all__ = [
    'Meter', 'NumericalMeter', 'HandyTimer',
    'AveragePrecisionMeter', 'DetectionAPMeter'
]

def div(numerator: Tensor, denom: Union[Tensor, int, float]) -> Tensor:
    """Handle division by zero"""
    if type(denom) in [int, float]:
        if denom == 0:
            return torch.zeros_like(numerator)
        else:
            return numerator / denom
    elif type(denom) is Tensor:
        zero_idx = torch.nonzero(denom == 0).squeeze(1)
        denom[zero_idx] += 1e-8
        return numerator / denom
    else:
        raise TypeError("Unsupported data type ", type(denom))

class Meter:
    """
    Base class
    """
    def __init__(self, maxlen: Optional[int] = None) -> None:
        self._deque = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def __len__(self) -> int:
        return len(self._deque)

    def __iter__(self) -> Iterable:
        return iter(self._deque)

    def __getitem__(self, i: int) -> Any:
        return self._deque[i]

    def __repr__(self) -> str:
        reprstr = self.__class__.__name__ + '('
        reprstr += 'maxlen='
        reprstr += str(self._maxlen)
        reprstr += ')'
        return reprstr

    def reset(self) -> None:
        """Reset the meter"""
        self._deque.clear()

    def append(self, x: Any) -> None:
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
    def items(self) -> List[Any]:
        """Return the content"""
        return [item for item in self._deque]

class NumericalMeter(Meter):
    """
    Meter class with numerals as elements
    """
    VALID_TYPES = [int, float]
    
    def __init__(self, maxlen: Optional[int] = None) -> None:
        super().__init__(maxlen=maxlen)

    def append(self, x: Union[int, float]) -> None:
        if type(x) in self.VALID_TYPES:
            super().append(x)
        else:
            raise TypeError("Given element \'{}\' is not a numeral".format(x))

    def sum(self) -> Union[int, float]:
        if len(self._deque):
            return sum(self._deque)
        else:
            raise ValueError("Cannot take sum. The meter is empty.")

    def mean(self) -> float:
        if len(self._deque):
            return sum(self._deque) / len(self._deque)
        else:
            raise ValueError("Cannot take mean. The meter is empty.")

    def max(self) -> Union[int, float]:
        if len(self._deque):
            return max(self._deque)
        else:
            raise ValueError("Cannot take max. The meter is empty.")

    def min(self) -> Union[int, float]:
        if len(self._deque):
            return min(self._deque)
        else:
            raise ValueError("Cannot take min. The meter is empty.")

class HandyTimer(NumericalMeter):
    """
    A timer class that tracks a sequence of time
    """
    def __init__(self, maxlen: Optional[int] = None):
        super().__init__(maxlen=maxlen)

    def __enter__(self) -> None:
        self._timestamp = time.time()

    def __exit__(self, type, value, traceback) -> None:
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
        precision(int, optional): Precision used for float-point operations. Choose
            amongst 64, 32 and 16. Default is 64
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
    def __init__(self, num_gt: Optional[Iterable] = None,
            algorithm: str = 'AUC', chunksize: int = -1,
            precision: int = 64,
            output: Optional[Tensor] = None,
            labels: Optional[Tensor] = None) -> None:
        self._dtype = eval('torch.float' + str(precision))
        self.num_gt = torch.as_tensor(num_gt, dtype=self._dtype) \
            if num_gt is not None else None
        self.algorithm = algorithm
        self._chunksize = chunksize
        
        is_none = (output is None, labels is None)
        if is_none == (True, True):
            self._output = torch.tensor([], dtype=self._dtype)
            self._labels = torch.tensor([], dtype=self._dtype)
        elif is_none == (False, False):
            self._output = output.detach().cpu().to(self._dtype)
            self._labels = labels.detach().cpu().to(self._dtype)
        else:
            raise AssertionError("Output and labels should both be given or None")

        self._output_temp = [torch.tensor([], dtype=self._dtype)]
        self._labels_temp = [torch.tensor([], dtype=self._dtype)]

    @staticmethod
    def compute_per_class_ap_as_auc(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Arguments: 
            tuple_(Tuple[Tensor, Tensor]): precision and recall
        Returns:
            ap(Tensor[1])
        """
        prec, rec = tuple_
        ap = 0
        max_rec = rec[-1]
        for idx in range(prec.numel()):
            # Stop when maximum recall is reached
            if rec[idx] >= max_rec:
                break
            d_x = rec[idx] - rec[idx - 1]
            # Skip when negative example is registered
            if d_x == 0:
                continue
            ap +=  prec[idx] * rec[idx] if idx == 0 \
                else 0.5 * (prec[idx] + prec[idx - 1]) * d_x
        return ap

    @staticmethod
    def compute_per_class_ap_with_interpolation(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Arguments:
            tuple_(Tuple[Tensor, Tensor]): precision and recall
        Returns:
            ap(Tensor[1])
        """
        prec, rec = tuple_
        ap = 0
        max_rec = rec[-1]
        for idx in range(prec.numel()):
            # Stop when maximum recall is reached
            if rec[idx] >= max_rec:
                break
            d_x = rec[idx] - rec[idx - 1]
            # Skip when negative example is registered
            if d_x == 0:
                continue
            # Compute interpolated precision
            max_ = prec[idx:].max()
            ap +=  max_ * rec[idx] if idx == 0 \
                else 0.5 * (max_ + torch.max(prec[idx - 1], max_)) * d_x
        return ap

    @staticmethod
    def compute_per_class_ap_with_11_point_interpolation(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Arguments:
            tuple_(Tuple[Tensor, Tensor]): precision and recall
        Returns:
            ap(Tensor[1])
        """
        prec, rec = tuple_
        dtype = rec.dtype
        ap = 0
        for t in torch.linspace(0, 1, 11, dtype=dtype):
            inds = torch.nonzero(rec >= t).squeeze()
            if inds.numel():
                ap += (prec[inds].max() / 11)
        return ap

    @classmethod            
    def compute_ap(cls, output: Tensor, labels: Tensor,
            num_gt: Optional[Tensor] = None,
            algorithm: str = 'AUC',
            chunksize: int = -1) -> Tensor:
        """
        Compute average precision under the classification setting. Scores of all 
        classes are retained for each sample.

        Arguments:
            output(Tensor[N, K])
            labels(Tensor[N, K])
            num_gt(Tensor[K]): Number of ground truth instances for each class
            algorithm(str): AP evaluation algorithm
            chunksize(int, optional): The approximate size the given iterable will be split
                into for each worker. Use -1 to make the argument adaptive to iterable size
                and number of workers
        Returns:
            ap(Tensor[K])
        """
        prec, rec = cls.compute_precision_and_recall(output, labels, 
            num_gt=num_gt)
        ap = torch.zeros(output.shape[1], dtype=prec.dtype)
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

        with multiprocessing.get_context('spawn').Pool() as pool:
            for idx, result in enumerate(pool.imap(
                func=algorithm_handle,
                # NOTE: Use transpose instead of T for compatibility
                iterable=zip(prec.transpose(0,1), rec.transpose(0,1)),
                chunksize=chunksize
            )):
                ap[idx] = result
        
        return ap

    @staticmethod
    def compute_precision_and_recall(output: Tensor, labels: Tensor,
            num_gt: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Arguments:
            output(Tensor[N, K])
            labels(Tensor[N, K])
            num_gt(Tensor[K])
        Returns:
            prec(Tensor[N, K])
            rec(Tensor[N, K])
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
        rec = div(tp, labels.sum(0)) if num_gt is None \
            else div(tp, num_gt)

        return prec, rec

    def append(self, output: Tensor, labels: Tensor) -> None:
        """
        Add new results to the meter

        Arguments:
            output(tensor[N, K]): Network output with N examples and K classes
            labels(tensor[N, K]): Binary labels
        """
        if isinstance(output, torch.Tensor) and isinstance(labels, torch.Tensor):
            assert output.shape == labels.shape, \
                "Output scores do not match the dimension of labelss"
            self._output_temp.append(output.detach().cpu().to(self._dtype))
            self._labels_temp.append(labels.detach().cpu().to(self._dtype))
        else:
            raise TypeError("Arguments should both be torch.Tensor")

    def reset(self, keep_old: bool = False) -> None:
        """
        Clear saved statistics

        Arguments:
            keep_tracked(bool): If True, clear only the newly collected statistics
                since last evaluation
        """
        if not keep_old:
            self._output = torch.tensor([], dtype=self._dtype)
            self._labels = torch.tensor([], dtype=self._dtype)
        self._output_temp = [torch.tensor([], dtype=self._dtype)]
        self._labels_temp = [torch.tensor([], dtype=self._dtype)]

    def eval(self) -> Tensor:
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
            self.num_gt = self.num_gt.to(dtype=self._labels.dtype)
            faulty_cls = torch.nonzero(self._labels.sum(0) > self.num_gt).squeeze(1)
            if len(faulty_cls):
                raise AssertionError("Class {}: ".format(faulty_cls.tolist())+
                    "Number of true positives larger than that of ground truth")
        if len(self._output) and len(self._labels):
            return self.compute_ap(self._output, self._labels, num_gt=self.num_gt,
                algorithm=self.algorithm, chunksize=self._chunksize)
        else:
            print("WARNING: Collected results are empty. "
                "Return zero AP for all class.")
            return torch.zeros(self._output.shape[1], dtype=self._dtype)

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
        nproc(int, optional): The number of processes used to compute mAP. Default: 20
        precision(int, optional): Precision used for float-point operations. Choose
            amongst 64, 32 and 16. Default is 64
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
    def __init__(self, num_cls: int, num_gt: Optional[Tensor] = None,
            algorithm: str = 'AUC', nproc: int = 20,
            precision: int = 64,
            output: Optional[List[Tensor]] = None,
            labels: Optional[List[Tensor]] = None) -> None:
        if num_gt is not None and len(num_gt) != num_cls:
            raise AssertionError("Provided ground truth instances"
                "do not have the same number of classes as specified")

        self.num_cls = num_cls
        self.num_gt = num_gt if num_gt is not None else \
            [None for _ in range(num_cls)]
        self.algorithm = algorithm
        self._nproc = nproc
        self._dtype = eval('torch.float' + str(precision))

        is_none = (output is None, labels is None)
        if is_none == (True, True):
            self._output = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
            self._labels = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
        elif is_none == (False, False):
            assert len(output) == len(labels), \
                "The given output does not have the same number of classes as labels"
            assert len(output) == num_cls, \
                "The number of classes in the given output does not match the argument"
            self._output = to_tensor(output, 
                input_format='list', dtype=self._dtype, device='cpu')
            self._labels = to_tensor(labels,
                input_format='list', dtype=self._dtype, device='cpu')
        else:
            raise AssertionError("Output and labels should both be given or None")

        self._output_temp = [[] for _ in range(num_cls)]
        self._labels_temp = [[] for _ in range(num_cls)]
    
    @classmethod
    def compute_ap(cls, output: List[Tensor], labels: List[Tensor],
            num_gt: Iterable, nproc: int, algorithm: str = 'AUC') -> Tuple[Tensor, Tensor]:
        """
        Compute average precision under the detection setting. Only scores of the 
        predicted classes are retained for each sample. As a result, different classes
        could have different number of predictions.

        Arguments:
            output(list[Tensor])
            labels(list[Tensor])
            num_gt(iterable): Number of ground truth instances for each class
            nproc(int, optional): The number of processes used to compute mAP
            algorithm(str): AP evaluation algorithm
        Returns:
            ap(Tensor[K])
            max_rec(Tensor[K])
        """
        ap = torch.zeros(len(output), dtype=output[0].dtype)
        max_rec = torch.zeros_like(ap)

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

        # Avoid multiprocessing when the number of processes is fewer than two
        if nproc < 2:
            for idx in range(len(output)):
                ap[idx], max_rec[idx] = cls.compute_ap_for_each((
                    idx, list(num_gt)[idx],
                    output[idx], labels[idx],
                    algorithm_handle
                ))
            return ap, max_rec

        with multiprocessing.get_context('spawn').Pool(nproc) as pool:
            for idx, results in enumerate(pool.map(
                func=cls.compute_ap_for_each,
                iterable=[(idx, ngt, out, gt, algorithm_handle) 
                    for idx, (ngt, out, gt) in enumerate(zip(num_gt, output, labels))]
            )):
                ap[idx], max_rec[idx] = results

        return ap, max_rec

    @classmethod
    def compute_ap_for_each(cls, tuple_):
        idx, num_gt, output, labels, algorithm = tuple_
        # Sanity check
        if num_gt is not None and labels.sum() > num_gt:
            raise AssertionError("Class {}: ".format(idx)+
                "Number of true positives larger than that of ground truth")
        if len(output) and len(labels):
            prec, rec = cls.compute_pr_for_each(output, labels, num_gt)
            return algorithm((prec, rec)), rec[-1]
        else:
            print("WARNING: Collected results are empty. "
                "Return zero AP for class {}.".format(idx))
            return 0, 0

    @staticmethod
    def compute_pr_for_each(output: Tensor, labels: Tensor,
            num_gt: Optional[Union[int, float]] = None) -> Tuple[Tensor, Tensor]:
        """
        Arguments:
            output(Tensor[N])
            labels(Tensor[N]): Binary labels for each sample
            num_gt(int or float): Number of ground truth instances
        Returns:
            prec(Tensor[N])
            rec(Tensor[N])
        """
        order = output.argsort(descending=True)

        tp = labels[order]
        fp = 1 - tp
        tp = tp.cumsum(0)
        fp = fp.cumsum(0)

        prec = tp / (tp + fp)
        rec = div(tp, labels.sum().item()) if num_gt is None \
            else div(tp, num_gt)

        return prec, rec

    def append(self, output: Tensor, prediction: Tensor, labels: Tensor) -> None:
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

    def reset(self, keep_old: bool = False) -> None:
        """
        Clear saved statistics

        Arguments:
            keep_tracked(bool): If True, clear only the newly collected statistics
                since last evaluation
        """
        num_cls = len(self._output_temp)
        if not keep_old:
            self._output = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
            self._labels = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
        self._output_temp = [[] for _ in range(num_cls)]
        self._labels_temp = [[] for _ in range(num_cls)]

    def eval(self) -> Tensor:
        """
        Evaluate the average precision based on collected statistics

        Returns:
            torch.Tensor[K]: Average precisions for K classes
        """
        self._output = [torch.cat([
            out1, torch.as_tensor(out2, dtype=self._dtype)
        ]) for out1, out2 in zip(self._output, self._output_temp)]
        self._labels = [torch.cat([
            tar1, torch.as_tensor(tar2, dtype=self._dtype)
        ]) for tar1, tar2 in zip(self._labels, self._labels_temp)]
        self.reset(keep_old=True)

        self.ap, self.max_rec = self.compute_ap(self._output, self._labels, self.num_gt,
            nproc=self._nproc, algorithm=self.algorithm)

        return self.ap
