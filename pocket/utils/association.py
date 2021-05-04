"""
Data association in detection tasks

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch

from typing import Tuple, Optional
from torch import FloatTensor, LongTensor

from ..ops import box_iou

class BoxAssociation:
    """
    Associate detection boxes with ground truth boxes

    Arguments:
        min_iou(float): The minimum intersection over union to identify a positive
        encoding(str): Encodings of the bounding boxes. Choose between 'coord' and 'pixel'
    """
    def __init__(self, min_iou: float, encoding: str = 'coord') -> None:
        self.min_iou = min_iou
        self.encoding = encoding

        self._max_iou = None
        self._max_idx = None

    @property
    def max_iou(self) -> FloatTensor:
        """Return the largest IoU with any ground truth instances for each detection"""
        if self._max_iou is None:
            raise NotImplementedError
        else:
            return self._max_iou
    @property
    def max_idx(self) -> LongTensor:
        """Return the index of ground truth instance each detection is associated with"""
        if self._max_idx is None:
            raise NotImplementedError
        else:
            return self._max_idx

    def _iou(self, boxes_1: FloatTensor, boxes_2: FloatTensor) -> FloatTensor:
        """Compute intersection over union"""
        return box_iou(boxes_1, boxes_2, encoding=self.encoding)

    def __call__(self,
        gt_boxes: FloatTensor,
        det_boxes: FloatTensor,
        scores: Optional[FloatTensor] = None
    ) -> FloatTensor:
        """
        Arguments:
            gt_boxes(FloatTensor[N, 4]): Ground truth bounding boxes in (x1, y1, x2, y2) format
            det_boxes(FloatTensor[M, 4]): Detected bounding boxes in (x1, y1, x2, y2) format
            scores(FloatTensor[M]): Confidence scores for each detection. If left as None, the
                highest IoU will be used to rank duplicated detections.
        Returns:
            labels(FloatTensor[M]): Binary labels indicating true positive or not
        """
        # Compute intersection over uion
        iou = self._iou(gt_boxes, det_boxes)

        max_iou, max_idx = iou.max(0)
        self._max_iou = max_iou
        self._max_idx = max_idx

        if scores is None:
            scores = max_iou

        # Assign each detection to the ground truth with highest IoU
        match = torch.zeros_like(iou)
        match[max_idx, torch.arange(iou.shape[1])] = max_iou

        match = match > self.min_iou

        labels = torch.zeros_like(scores)
        # Determine true positives
        for i, m in enumerate(match):
            match_idx = torch.nonzero(m).squeeze(1)
            if len(match_idx) == 0:
                continue
            match_scores = scores[match_idx]
            labels[match_idx[match_scores.argmax()]] = 1

        return labels

class BoxPairAssociation(BoxAssociation):
    """
    Associate detection box pairs with ground truth box pairs

    Arguments:
        min_iou(float): The minimum intersection over union to identify a positive
        encoding(str): Encodings of the bounding boxes. Choose between 'coord' and 'pixel'
    """
    def __init__(self, min_iou: float, encoding: str = 'coord') -> None:
        super().__init__(min_iou, encoding)

    def _iou(self,
            boxes_1: Tuple[FloatTensor, FloatTensor],
            boxes_2: Tuple[FloatTensor, FloatTensor]) -> FloatTensor:
        """
        Override method to compute IoU for box pairs

        Arguments:
            boxes_1(tuple): Ground truth box pairs in a 2-tuple
            boxes_2(tuple): Detection box pairs in a 2-tuple
        """
        return torch.min(
            box_iou(boxes_1[0], boxes_2[0], encoding=self.encoding),
            box_iou(boxes_1[1], boxes_2[1], encoding=self.encoding)
        )