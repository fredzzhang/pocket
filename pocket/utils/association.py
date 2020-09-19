"""
Data association in detection tasks

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch

from torchvision.ops.boxes import box_iou

class BoxAssociation:
    def __init__(self, min_iou):
        self.min_iou = min_iou

    def __call__(self, boxes_1, boxes_2, scores):
        """

        Arguments:
            boxes_1(FloatTensor[N, 4]): Ground truth bounding boxes
            boxes_2(FloatTensor[M, 4]): Detected bounding boxes
            scores(FloatTensor[M]): Confidence scores for each detection
        Returns:
            labels(FloatTensor[M])
        """
        # Compute intersection over uion
        iou = box_iou(boxes_1, boxes_2)

        # Assign each detection to the ground truth with highest IoU
        max_iou, max_idx = iou.max(0)
        match = -1 * torch.ones_like(iou)
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