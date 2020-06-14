"""
Visualisation utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import numpy as np

from PIL import Image, ImageDraw

def draw_boxes(image, boxes, fill=None, outline=None):
    """Draw bounding boxes onto a PIL image

    Arguments:
        image(PIL Image)
        boxes(torch.Tensor[N,4] or np.ndarray[N,4] or List[List]): Bounding box
            coordinates in the format (x1, y1, x2, y2)
        fill(str)
        outline(str)
    """
    if isinstance(boxes, (torch.Tensor, np.ndarray)):
        boxes = boxes.tolist()
    elif not isinstance(boxes, list):
        raise TypeError("Bounding box coords. should be torch.Tensor, np.ndarray or list")

    canvas = ImageDraw.Draw(image)
    for box in boxes:
        canvas.rectangle(box, fill=fill, outline=outline)
    return image