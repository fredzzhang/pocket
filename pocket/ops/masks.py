"""
Mask utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
from ..cpp import generate_masks

def generate_binary_masks(boxes, h, w):
    """
    Generate binary masks based on bounding box coordinates. Pixels completely
    included in a bounding box are assigned values of 1. Those that are partially
    included are assigned values equal to the overlapping area (<1). Pixels outside
    of the bounding boxes are assigned values of 0.

    Arguments:
        boxes(Tensor[N, 4]): Bounding box coordinates (x1, y1, x2, y2)
        h(int): Height of the mask
        w(int): Width of the mask
    Returns:
        Tensor[N, h, w]
    """
    if not isinstance(boxes, torch.Tensor):
        raise AssertionError("Provided bounding boxes are not instances of Tensor")
    if not isinstance(h, int) or not isinstance(w, int):
        raise AssertionError("Image width and height should both of integers")

    boxes = boxes.float()
    return generate_masks(boxes, h, w)