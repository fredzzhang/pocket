"""
Utilities related to masked RoI align

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
from torchvision.ops.poolers import LevelMapper
from torchvision.ops._utils import convert_boxes_to_roi_format
from torchvision.ops.roi_align import _RoIAlignFunction

def masked_roi_align(features, boxes, masks, output_size,
        spatial_scale=1.0, sampling_ratio=-1, clone_limit=512):
    """
    Perform masked RoI align given individual bounding boxes and corresponding masks

    Arguments:
        features(Tensor[N, C, H, W]): Input feature tensor
        boxes(list[Tensor[M, 4]] or Tensor[K, 5]): The box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from. If a single Tensor is passed,
            then the first column should contain the batch index. If a list of Tensors
            is passed, then each Tensor will correspond to the boxes for an element i
            in a batch
        masks(list[Tensor[M, C, H, W]] or Tensor[K, C, H, W]): The masks to be applied on
            feature maps for each bounding box. 
        output_size(int or Tuple[int, int]): The size of the output after the cropping
            is performed, as (height, width)
        spatial_scale(float): A scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0
        sampling_ratio(int): Number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly sampling_ratio x sampling_ratio grid points are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ceil(roi_width / pooled_w), and likewise for height). Default: -1
        clone_limit(int): The maximum number of feature map clones set for memory
            and speed concerns. Default: 512
    """
    if type(output_size) is int:
        output_size = (output_size, output_size)
    if not isinstance(boxes, torch.Tensor):
        boxes = convert_boxes_to_roi_format(boxes)
    if not isinstance(masks, torch.Tensor):
        masks = torch.cat(masks, 0)

    num_boxes = len(boxes)
    num_iter = num_boxes // clone_limit + bool(num_boxes % clone_limit)
    output = torch.zeros(num_boxes, features.shape[1], *output_size,
        device=features.device)

    for idx in range(num_iter):
        start_idx = idx * clone_limit
        end_idx = min(start_idx + clone_limit, num_boxes)

        per_instance_features = features[
            boxes[start_idx: end_idx, 0].long(), :, :, :]
        # Modify the batch index to align with feature map clones
        boxes[start_idx: end_idx, 0] = torch.arange(end_idx - start_idx,
            device=boxes.device, dtype=boxes.dtype)
        output[start_idx: end_idx, :, :, :] = \
            _RoIAlignFunction.apply(
                per_instance_features * masks[start_idx: end_idx, :, :, :],
                boxes[start_idx: end_idx, :],
                output_size,
                spatial_scale,
                sampling_ratio
            )

    return output

class MaskedRoIAlign(torch.nn.Module):
    """
    Masked RoI align
    """
    def __init__(self, output_size, spatial_scale, sampling_ratio, clone_limit):
        super().__init__()
        self.output_size = (output_size, output_size) if type(output_size) is int \
            else output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.clone_limit = clone_limit

    def forward(self, features, boxes, masks):
        return masked_roi_align(features, boxes, masks,
            self.output_size, self.spatial_scale, self.sampling_ratio, self.clone_limit)

    def __repr__(self):
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '('
        reprstr += 'output_size='
        reprstr += repr(self.output_size)
        reprstr += ', spatial_scale='
        reprstr += repr(self.spatial_scale)
        reprstr += ', sampling_ratio='
        reprstr += repr(self.sampling_ratio)
        reprstr += ', clone_limit='
        reprstr += repr(self.clone_limit)
        reprstr += ')'
        return reprstr

class BoxPairMultiScaleRoIAlign(torch.nn.Module):
    """
    """
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        pass