"""
Operations related to masked RoI align

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

"""
Acknowledgement:

Source code in this module is largely modified from
https://github.com/pytorch/vision/tree/master/torchvision/ops/roi_align.py

See below for detailed license

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
from torchvision.ops._utils import convert_boxes_to_roi_format
from torchvision.ops.roi_align import _RoIAlignFunction

def masked_roi_align(features, boxes, masks, output_size,
        spatial_scale=1.0, sampling_ratio=-1, 
        mem_limit=8, reserve=128):
    """
    Perform masked RoI align given individual bounding boxes and corresponding masks

    The function makes a copy of the corresponding feature map for each box and subsequently
    applies a mask. To avoid memory overflow, the maximum number of copies made is inferred
    from given arguments {mem_limit} and {reserve}. Inappropiate choice of arguments could 
    lead to memory overflow.

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
        mem_limit(int): Memory limit (GB) allowed in this module. The maximum number of feature
            map clones made will be inferred from this. Default: 8
        reserve(int): Memory (MB) overhead preserved for miscellaneous variables. The memory
            limit will be subtracted by this value. Default: 128
    """
    if type(output_size) is int:
        output_size = (output_size, output_size)
    if not isinstance(boxes, torch.Tensor):
        boxes = convert_boxes_to_roi_format(boxes)
    if not isinstance(masks, torch.Tensor):
        masks = torch.cat(masks, 0)

    num_boxes = len(boxes)
    output_shape = (num_boxes, features.shape[1],) + output_size
    output = []

    MB = 1024 ** 2; GB = MB * 1024
    # The memory available for cloning feature maps
    clone_limit = ((
        mem_limit * GB - reserve * MB
        - torch.as_tensor(output_shape).prod() * 4
        - torch.as_tensor(masks.shape).prod() * 4
        ) / torch.as_tensor(features.shape[1:]).prod() * 4
    ).item()
        
    num_iter = num_boxes // clone_limit + bool(num_boxes % clone_limit)
    # Compute pooled features iteratively based on maximum number of feature map
    # clones allowed
    for idx in range(num_iter):
        start_idx = idx * clone_limit
        end_idx = min(start_idx + clone_limit, num_boxes)

        per_instance_features = features[
            boxes[start_idx: end_idx, 0].long()]
        # Modify the batch index to align with feature map clones
        boxes[start_idx: end_idx, 0] = torch.arange(end_idx - start_idx,
            device=boxes.device, dtype=boxes.dtype)
        output.append(
            _RoIAlignFunction.apply(
                per_instance_features * masks[start_idx: end_idx],
                boxes[start_idx: end_idx, :],
                output_size,
                spatial_scale,
                sampling_ratio
            )
        )
    output = torch.cat(output, 0)
    assert output.shape == output_shape, \
        "Inconsistent feature size"

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