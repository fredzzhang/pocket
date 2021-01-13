"""
Useful transforms 

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import random
import torchvision

__all__ = [
    'to_tensor', 'horizontal_flip_boxes', 'horizontal_flip',
    'ToTensor', 'RandomHorizontalFlip', 'Flatten'
]

def _to_list_of_tensor(x, dtype=None, device=None):
    return [torch.as_tensor(item, dtype=dtype, device=device) for item in x]

def _to_tuple_of_tensor(x, dtype=None, device=None):
    return tuple(torch.as_tensor(item, dtype=dtype, device=device) for item in x)

def _to_dict_of_tensor(x, dtype=None, device=None):
    return dict([(k, torch.as_tensor(v, dtype=dtype, device=device)) for k, v in x.items()])

def to_tensor(x, input_format='tensor', dtype=None, device=None):
    """Convert input data to tensor based on its format"""
    if input_format == 'tensor':
        return torch.as_tensor(x, dtype=dtype, device=device)
    elif input_format == 'pil':
        return torchvision.transforms.functional.to_tensor(x).to(
            dtype=dtype, device=device)
    elif input_format == 'list':
        return _to_list_of_tensor(x, dtype=dtype, device=device)
    elif input_format == 'tuple':
        return _to_tuple_of_tensor(x, dtype=dtype, device=device)
    elif input_format == 'dict':
        return _to_dict_of_tensor(x, dtype=dtype, device=device)
    else:
        raise ValueError("Unsupported format {}".format(input_format))

def horizontal_flip_boxes(w, boxes, encoding='coords'):
    """
    Horizontally flip bounding boxes with respect to an image.

    Parameters
    ----------
    w : int
        Width of the image, to which given boxes belong.
    boxes : Tensor
        Bounding box tensors (N, 4) in the format x1, y1, x2, y2.
    encoding : str, optional
        Encoding method of the bounding boxes. Choose between 'coords' and 'pixel'.
        Using 'coords' implies the boxes are encoded as coordinates, which mean the
        values are no more than the dimensions of the image. Using 'pixel' implies
        the boxes are encoded as pixel indices, meaning the largest values are no
        more than the dimensions of the image minus one (Default is 'coords').

    Returns
    -------
    boxes: Tensor
        Flipped bounding box tensors (N, 4) in the same format and encoding.
    """
    boxes = boxes.clone()
    if encoding == 'coords':
        x_min = w - boxes[:, 2]
        x_max = w - boxes[:, 0]
        boxes[:, 0] = torch.clamp(x_min, 0, w)
        boxes[:, 2] = torch.clamp(x_max, 0, w)
    elif encoding == 'pixel':
        x_min = w - boxes[:, 2] - 1
        x_max = w - boxes[:, 0] - 1
        boxes[:, 0] = torch.clamp(x_min, 0, w - 1)
        boxes[:, 2] = torch.clamp(x_max, 0, w - 1)
    else:
        raise ValueError("Unknown box encoding \'{}\'".format(encoding))
    return boxes

def horizontal_flip(image, boxes=None, encoding='coords'):
    """
    Horizontally flip an image and its associated bounding boxes if any.

    Parameters
    ----------
    image : PIL.Image
        Input image.
    boxes : Tensor, optional
        Bounding box tensors (N, 4) in the format x1, y1, x2, y2. When left as None
        only the image will be flipped (Default is None).
    encoding : str, optional
        Encoding method of the bounding boxes. Choose between 'coords' and 'pixel'.
        Using 'coords' implies the boxes are encoded as coordinates, which mean the
        values are no more than the dimensions of the image. Using 'pixel' implies
        the boxes are encoded as pixel indices, meaning the largest values are no
        more than the dimensions of the image minus one (Default is 'coords').

    Returns
    -------
    image: PIL.Image
        Input image horizontally flipped.
    boxes: Tensor
        Flipped bounding box tensors (N, 4) in the same format and encoding. This will
        only be returned when the input argument `boxes` is not None.
    """
    image = torchvision.transforms.functional.hflip(image)
    w, _ = image.size
    if boxes is None:
        return image
    else:
        boxes = horizontal_flip_boxes(w, boxes, encoding=encoding)
    return image, boxes

class ToTensor:
    """Convert to tensor"""
    def __init__(self, input_format='tensor', dtype=None, device=None):
        self.input_format = input_format
        self.dtype = dtype
        self.device = device
    def __call__(self, x):
        return to_tensor(x, 
            input_format=self.input_format,
            dtype=self.dtype,
            device=self.device
        )
    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += 'input_format={}'.format(repr(self.input_format))
        reprstr += ', dtype='
        reprstr += repr(self.dtype)
        reprstr += ', device='
        reprstr += repr(self.device)
        reprstr += ')'
        return reprstr

class RandomHorizontalFlip:
    """Horizontally flip an image and its associated bounding boxes if any"""
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, image, boxes=None, encoding='coords'):
        prob = random.random()
        if prob < self.prob and boxes is None:
            return image
        elif prob < self.prob and boxes is not None:
            return image, boxes
        return horizontal_flip(image, boxes, encoding)

class Flatten(torch.nn.Module):
    """Flatten a tensor"""
    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    def forward(self, x):
        return x.flatten(start_dim=self.start_dim, end_dim=self.end_dim)
