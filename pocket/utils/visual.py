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
        boxes(torch.Tensor[N,4] or np.ndarray[N,4] or List[List[4]]): Bounding box
            coordinates in the format (x1, y1, x2, y2)
        kwargs(dict): Parameters for PIL.ImageDraw.Draw.rectangle
    """
    if isinstance(boxes, (torch.Tensor, list)):
        boxes = np.asarray(boxes)
    elif not isinstance(boxes, np.ndarray):
        raise TypeError("Bounding box coords. should be torch.Tensor, np.ndarray or list")
    boxes = boxes.reshape(-1, 4).tolist()

    canvas = ImageDraw.Draw(image)
    for box in boxes:
        canvas.rectangle(box, **kwargs)

def draw_box_pairs(image, boxes_1, boxes_2, width=1):
    """Draw bounding box pairs onto a PIL image. Boxes corresponding to argument <boxes_1>
    are drawn in blue, and green for the other group. Corresponding box pairs will be joined
    by a red line connecting the centres of the boxes

    Arguments:
        image(PIL Image)
        boxes_1(torch.Tensor[N,4] or np.ndarray[N,4] or List[List[4]]): Bounding box
            coordinates in the format (x1, y1, x2, y2)
        boxes_2: Same format as above
        width: Width of the boxes
    """
    if isinstance(boxes_1, (torch.Tensor, list)):
        boxes_1 = np.asarray(boxes_1)
    elif not isinstance(boxes_1, np.ndarray):
        raise TypeError("Bounding box coords. should be torch.Tensor, np.ndarray or list")
    if isinstance(boxes_2, (torch.Tensor, list)):
        boxes_2 = np.asarray(boxes_2)
    elif not isinstance(boxes_2, np.ndarray):
        raise TypeError("Bounding box coords. should be torch.Tensor, np.ndarray or list")
    boxes_1 = boxes_1.reshape(-1, 4)
    boxes_2 = boxes_2.reshape(-1, 4)

    canvas = ImageDraw.Draw(image)
    assert len(boxes_1) == len(boxes_2), "Number of boxes does not match between two given groups"
    for b1, b2 in zip(boxes_1, boxes_2):
        canvas.rectangle(b1.tolist(), outline='#007CFF', width=width)
        canvas.rectangle(b2.tolist(), outline='#46FF00', width=width)
        b_h_centre = (b1[:2]+b1[2:])/2
        b_o_centre = (b2[:2]+b2[2:])/2
        canvas.line(
            b_h_centre.tolist() + b_o_centre.tolist(),
            fill='#FF4444', width=width
        )
        canvas.ellipse(
            (b_h_centre - width).tolist() + (b_h_centre + width).tolist(),
            fill='#FF4444'
        )
        canvas.ellipse(
            (b_o_centre - width).tolist() + (b_o_centre + width).tolist(),
            fill='#FF4444'
        )


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
    num = int(hyp / length)

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
