"""
Bounding box operations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torchvision.ops.boxes as box_ops

Tensor = torch.Tensor

def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Perform non-maximum suppression (NMS) on boxes

    Parameters:
    -----------
    boxes: torch.Tensor
        Bounding box coordinates formatted as [[x1, y1, x2, y2],...]
    scores: torch.Tensor
        Scores for bounding boxes
    iou_threshold: float
        IoU threshold for identify overlapping boxes

    Returns:
    --------
    keep: torch.Tensor
        Indices of kept boxes
    """
    criteria = box_ops.box_iou(boxes, boxes) >= iou_threshold
    active = scores.argsort(descending=True)
    keep = []
    while len(active):
        i = active[0]
        keep.append(i.item())
        rm = torch.nonzero(criteria[i]).squeeze(1)
        active = [k for k in active if k not in rm]
    keep = torch.as_tensor(keep, device=boxes.device)
    return keep

def pnms(boxes_1: Tensor, boxes_2: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Perform non-maximum suppression (NMS) on box pairs

    Parameters:
    -----------
    boxes_1: torch.Tensor
        Bounding box coordinates formatted as [[x1, y1, x2, y2],...]
    boxes_2: torch.Tensor
        Bounding box coordinates formatted as [[x1, y1, x2, y2],...]
    scores: torch.Tensor
        Scores for bounding boxes
    iou_threshold: float
        IoU threshold for identify overlapping boxes

    Returns:
    --------
    keep: torch.Tensor
        Indices of kept box pairs
    """
    criteria = torch.min(
        box_ops.box_iou(boxes_1, boxes_1),
        box_ops.box_iou(boxes_2, boxes_2)
    ) >= iou_threshold
    active = scores.argsort(descending=True)
    keep = []
    while len(active):
        i = active[0]
        keep.append(i.item())
        rm = torch.nonzero(criteria[i]).squeeze(1)
        active = [k for k in active if k not in rm]
    keep = torch.as_tensor(keep, device=boxes_1.device)
    return keep

def batched_pnms(
    boxes_1: Tensor, boxes_2: Tensor,
    scores: Tensor, classes: Tensor,
    iou_threshold: float
) -> Tensor:
    """
    Perform non-maximum suppression (NMS) on box pairs from the same class.

    Parameters:
    -----------
    boxes_1: torch.Tensor
        Bounding box coordinates formatted as [[x1, y1, x2, y2],...]
    boxes_2: torch.Tensor
        Bounding box coordinates formatted as [[x1, y1, x2, y2],...]
    scores: torch.Tensor
        Scores for bounding boxes
    iou_threshold: float
        IoU threshold for identify overlapping boxes

    Returns:
    --------
    keep: torch.Tensor
        Indices of kept box pairs
    """
    if boxes_1.numel() == 0 or boxes_2.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes_1.device)
    else:
        max_coordinate = torch.max(boxes_1, boxes_2).max()
        offsets = classes.to(boxes_1) * (max_coordinate + torch.tensor(1).to(boxes_1))
        bx1_for_nms = boxes_1 + offsets[:, None]
        bx2_for_nms = boxes_2 + offsets[:, None]
        keep = pnms(bx1_for_nms, bx2_for_nms, scores, iou_threshold)
        return keep

def box_iou(boxes_1: Tensor, boxes_2: Tensor, encoding: str = 'coord') -> Tensor:
    """
    Compute intersection over union between boxes

    Parameters:
    -----------
    boxes_1: torch.Tensor
        (N, 4) Bounding boxes formatted as [[x1, y1, x2, y2],...]
    boxes_2: torch.Tensor
        (M, 4) Bounding boxes formatted as [[x1, y1, x2, y2],...]
    encoding: str
        A string that indicates what the boxes encode
            'coord': Coordinates of the two corners
            'pixel': Pixel indices of the two corners

    Returns:
    --------
    torch.Tensor
        Intersection over union of size (N, M)

    """
    if encoding == 'coord':
        return box_ops.box_iou(boxes_1, boxes_2)
    elif encoding == 'pixel':
        w1 = (boxes_1[:, 2] - boxes_1[:, 0] + 1).clamp(min=0)
        h1 = (boxes_1[:, 3] - boxes_1[:, 1] + 1).clamp(min=0)
        s1 = w1 * h1
        w2 = (boxes_2[:, 2] - boxes_2[:, 0] + 1).clamp(min=0)
        h2 = (boxes_2[:, 3] - boxes_2[:, 1] + 1).clamp(min=0)
        s2 = w2 * h2

        n1 = len(boxes_1); n2 = len(boxes_2)
        i, j = torch.meshgrid(
            torch.arange(n1),
            torch.arange(n2)
        )
        i = i.flatten(); j = j.flatten()
        
        x1, y1 = torch.max(boxes_1[i, :2], boxes_2[j, :2]).unbind(1)
        x2, y2 = torch.min(boxes_1[i, 2:], boxes_2[j, 2:]).unbind(1)
        w_intr = (x2 - x1 + 1).clamp(min=0)
        h_intr = (y2 - y1 + 1).clamp(min=0)
        s_intr = w_intr * h_intr

        iou = s_intr / (s1[i] + s2[j] - s_intr)
        return iou.reshape(n1, n2)
    else:
        raise ValueError("The encoding type should be either \"coord\" or \"pixel\"")

def box_giou(boxes_1: Tensor, boxes_2: Tensor) -> Tensor:
    """
    Compute generalised intersection over union between boxes

    Reference: Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression
    https://arxiv.org/abs/1902.09630

    Parameters:
    -----------
    boxes_1: torch.Tensor
        Bounding boxes of size (N, 4) formatted as [[x1, y1, x2, y2],...]
    boxes_2: torch.Tensor
        Bounding boxes of size (M, 4) formatted as [[x1, y1, x2, y2],...]

    Returns:
    --------
    giou: torch.Tensor
        Generalised IoUs of size (N, M)
    """
    i, j = torch.meshgrid(
        torch.arange(len(boxes_1)),
        torch.arange(len(boxes_2))
    )
    boxes_1 = boxes_1[i]
    boxes_2 = boxes_2[j]

    x1min, y1min, x2min, y2min = torch.min(boxes_1, boxes_2).unbind(-1)
    x1max, y1max, x2max, y2max = torch.max(boxes_1, boxes_2).unbind(-1)

    intersection = (x2min - x1max).clamp(min=0) * (y2min - y1max).clamp(min=0)
    convex_hull = (x2max - x1min) * (y2max - y1min)
    union = torch.prod(
        boxes_1[..., 2:] - boxes_1[..., :2], dim=-1
    ) + torch.prod(
        boxes_2[..., 2:] - boxes_2[..., :2], dim=-1
    ) - intersection

    iou = intersection / union
    giou = iou - (convex_hull - union) / convex_hull

    return giou