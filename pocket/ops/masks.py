"""
Mask utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
try:
    from .. import cpp
    CPP_COMPILED = True
except:
    CPP_COMPILED = False

__all__ = [
    'generate_masks'
]

def generate_masks(boxes, h, w):
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
    if not CPP_COMPILED:
        raise NotImplementedError

    if not isinstance(boxes, torch.Tensor):
        raise AssertionError("Provided bounding boxes are not instances of Tensor")
    if not isinstance(h, int) or not isinstance(w, int):
        raise AssertionError("Image width and height should both of integers")
    
    if len(boxes):
        assert torch.all(boxes >= 0)
        assert boxes[:, 2].max() <= w
        assert boxes[:, 3].max() <= h
    else:
        return torch.empty(0, h, w)

    boxes = boxes.float()

    return cpp.generate_masks(boxes, h, w)
