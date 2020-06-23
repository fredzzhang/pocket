"""
Visualisation utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import numpy as np

from PIL import Image, ImageDraw

def draw_boxes(image, boxes, **kwargs):
    """Draw bounding boxes onto a PIL image

    Arguments:
        image(PIL Image)
        boxes(torch.Tensor[N,4] or np.ndarray[N,4] or List[List]): Bounding box
            coordinates in the format (x1, y1, x2, y2)
        kwargs(dict): Parameters for PIL.ImageDraw.Draw.rectangle
    """
    if isinstance(boxes, (torch.Tensor, np.ndarray)):
        boxes = boxes.tolist()
    elif not isinstance(boxes, list):
        raise TypeError("Bounding box coords. should be torch.Tensor, np.ndarray or list")

    canvas = ImageDraw.Draw(image)
    for box in boxes:
        canvas.rectangle(box, **kwargs)

def draw_dashed_line(image, xy, length=5, **kwargs):
    """Draw dashed lines onto a PIL image

    Arguments:
        image(PIL Image)
        xy(torch.Tensor[4] or np.ndarray[4] or List[4]): [x1, y1, x2, y2]
        length(int): Length of line segments
    """
    if isinstance(xy, torch.Tensor):
        xy = xy.numpy()
    elif isinstance(xy, list):
        xy = np.asarray(xy)
    elif not isinstance(xy, np.ndarray):
        raise TypeError("Point coords. should be torch.Tensor, np.ndarray or list")

    canvas = ImageDraw.Draw(image)
    w = xy[2] - xy[0]; h = xy[3] - xy[1]
    hyp = np.sqrt(w ** 2 + h ** 2)
    num = hyp / length

    xx = np.linspace(xy[0], xy[2], num=num)
    yy = np.linspace(xy[1], xy[3], num=num)

    for i in range(int(len(xx) / 2)):
        canvas.line((
            xx[i * 2], yy[i * 2],
            xx[i * 2 + 1], yy[i * 2 + 1]
            ), **kwargs
        )

def draw_dashed_rectangle(image, xy, **kwargs):
    """Draw rectangle in dashed lines"""
    if isinstance(xy, torch.Tensor):
        xy = xy.numpy()
    elif isinstance(xy, list):
        xy = np.asarray(xy)
    elif not isinstance(xy, np.ndarray):
        raise TypeError("Point coords. should be torch.Tensor, np.ndarray or list")

    xy_ = xy.copy(); xy_[3] = xy_[1]
    draw_dashed_line(image, xy_, **kwargs)
    xy_ = xy.copy(); xy_[0] = xy_[2]
    draw_dashed_line(image, xy_, **kwargs)
    xy_ = xy.copy(); xy_[1] = xy_[3]
    draw_dashed_line(image, xy_, **kwargs)
    xy_ = xy.copy(); xy_[2] = xy_[0]
    draw_dashed_line(image, xy_, **kwargs)