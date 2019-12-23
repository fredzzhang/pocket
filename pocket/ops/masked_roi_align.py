"""
"""

import torch
from torchvision.ops.poolers import LevelMapper
from torchvision.ops._utils import convert_boxes_to_roi_format
from torchvision.ops.roi_align import _RoIAlignFunction

def masked_roi_align(features, boxes, masks, output_size,
        spatial_scale=1.0, sampling_ratio=-1, batch_limit=512):
    """
    features(Tensor[N, C, H, W])
    boxes(list[Tensor[M, 4]] or Tensor[K, 5])
    masks(list[Tensor[M, C, H, W]] or Tensor[K, C, H, W])
    output_size(int or Tuple[int, int])
    spatial_scale(float)
    sampling_ratio(int)
    batch_limit(int): The maximum number of clones of a feature map
    """
    if type(output_size) is int:
        output_size = (output_size, output_size)
    if not isinstance(boxes, torch.Tensor):
        boxes = convert_boxes_to_roi_format(boxes)
    if not isinstance(masks, torch.Tensor):
        masks = torch.cat(masks, 0)

    num_boxes = len(boxes)
    num_iter = num_boxes // batch_limit + bool(num_boxes % batch_limit)
    output = torch.zeros(num_boxes, features[1], *output_size,
        device=features.device)

    for idx in range(num_iter):
        start_idx = idx * batch_limit
        end_idx = min(start_idx + batch_limit, num_boxes)

        per_instance_features = features[boxes[start_idx: end_idx, 0], :, :, :]
        boxes[start_idx: end_idx, 0] = torch.arange(end_idx - start_idx,
            device=boxes.device, dtype=boxes.dtype)
        output[start_idx: end_idx, :, :, :] = \
            _RoIAlignFunction(
                per_instance_features * masks[start_idx: end_idx, :, :, :],
                boxes[start_idx: end_idx, :],
                output_size,
                spatial_scale,
                sampling_ratio
            )

    return output

class MaskedMultiScaleRoIAlign(torch.nn.Module):
    """
    """
    def __init__(self, featmap_names, output_size, sampling_ratio):
        pass