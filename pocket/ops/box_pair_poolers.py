"""
Operations related to box pair pooling

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

"""
Acknowledgement:

Source code in this module is largely modified from
https://github.com/pytorch/vision/blob/master/torchvision/ops/poolers.py

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
from torchvision.ops.poolers import LevelMapper
from torchvision.ops.boxes import clip_boxes_to_image
from torchvision.ops._utils import convert_boxes_to_roi_format

from .masked_roi_align import masked_roi_align

__all__ = [
    'BoxPairMultiScaleRoIAlign'
]

class LevelMapper_(LevelMapper):
    """Modify torchvision.ops.poolers.LevelMapper"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __call__(self, boxes_1, boxes_2=None):
        """
        Override __call__ method for different input format

        Arguments:
            boxes_1(Tensor[N, 5])
            boxes_2(Tensor[N, 5])
        """
        boxes_2 = boxes_1.clone() if boxes_2 is None else boxes_2
        # Use the smaller area between the two groups of boxes
        s = torch.min(
            torch.sqrt((boxes_1[:, 3] - boxes_1[:, 1]) * (boxes_1[:, 4] - boxes_1[:, 2])),
            torch.sqrt((boxes_2[:, 3] - boxes_2[:, 1]) * (boxes_2[:, 4] - boxes_2[:, 2]))
        )
        # Copied from torchvision.ops.poolers.LevelMapper
        # as per Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min

class BoxPairMultiScaleRoIAlign(torch.nn.Module):
    """
    Multi scale RoI align for box pairs

    Arguments:
        output_size(int or Tuple[int, int]): The size of the output after the cropping
            is performed, as (height, width)
        spatial_scale(Tuple[float]): Scaling factors between the bounding box coordinates
            and feature coordinates. When provided features spread across multiple levels,
            the tuple of scales are expected to be organized in descending order. And each
            scale is expected to be in the format of 2 ^ (-k). The number of scales given
            will indicate the number of feature maps
        sampling_ratio(int): Number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly sampling_ratio x sampling_ratio grid points are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ceil(roi_width / pooled_w), and likewise for height)
        mask_limit(int): Maximum number of masks produced at once for memory concerns
    """
    def __init__(self, output_size, spatial_scale, sampling_ratio, mask_limit=512):
        super().__init__()
        self.output_size = (output_size, output_size) if type(output_size) is int \
            else output_size
        self.num_levels = len(spatial_scale)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.mask_limit = mask_limit

        lvl_min = -torch.log2(torch.tensor(max(spatial_scale),
            dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(min(spatial_scale),
            dtype=torch.float32)).item()
        self.map_levels = LevelMapper_(lvl_min, lvl_max)

    def __repr__(self):
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '('
        reprstr += 'output_size='
        reprstr += repr(self.output_size)
        reprstr += ', spatial_scale='
        reprstr += repr(self.spatial_scale)
        reprstr += ', sampling_ratio='
        reprstr += repr(self.sampling_ratio)
        reprstr += ', mask_limit='
        reprstr += repr(self.mask_limit)
        reprstr += ')'
        return reprstr

    def fill_masks_for_boxes(self, mask, box):
        """
        Fill an empty mask based on scaled box coordinates

        Arguments:
            mask(Tensor[C, H, W])
            box(Tensor[4]): Box coordinates aligned with mask scale
        Returns:
            mask(Tensor[C, H, W])

        NOTE: This following crucial to understand

        1. For a pixel, the coordinates of the point at its top left corner
            give its index
        2. For a point with non-integer coordinates (x,y), the index of the
            pixel that contains the point is always (x.floor(), y.floor())
        3. For a bounding box defined as (x1, y1, x2, y2), the pixel that contains
            the top left corner can be indexed by [x1.floor(), y1.floor()], but 
            the pixel that contains the bottom right coner must be indexed by
            [x2.ceil()-1, y2.ceil()-1], NOT [x2.floor(), y2.floor()]
        """

        mask = mask.clone()
        # Expand the scaled bounding box to integer coordinates
        mask[:,
            box[1].floor().long(): box[3].ceil().long(),
            box[0].floor().long(): box[2].ceil().long(),
        ] = 1
        # Now attenuate the mask values at the expanded pixels
        mask[:,
            box[1].floor().long(): box[3].ceil().long(),
            box[0].floor().long(),
        ] *= (1 + box[0].floor() - box[0])
        mask[:,
            box[1].floor().long(): box[3].ceil().long(),
            box[2].ceil().long() - 1,
        ] *= (1 + box[2] - box[2].ceil())
        mask[:,
            box[1].floor().long(),
            box[0].floor().long(): box[2].ceil().long(),
        ] *= (1 + box[1].floor() - box[1])
        mask[:,
            box[3].ceil().long() - 1,
            box[0].floor().long(): box[2].ceil().long(),
        ] *= (1 + box[3] - box[3].ceil())

        return mask

    def construct_masks_for_box_pairs(self, features, level, boxes_h, boxes_o):
        """
        Arguments:
            features(Tensor[N, C, H, W])
            level(int): Level index of the given features. Should be between
                0 and (lvl_max - lvl_min)
            boxes_h(Tensor[K, 5])
            boxes_o(Tensor[K, 5])
        Returns:
            masks(Tensor[K, C, H, W])
        """
        masks = torch.zeros_like(features)[boxes_h[:, 0].long()]
        scale = self.spatial_scale[level]

        boxes_h[:, 1:] *= scale
        boxes_o[:, 1:] *= scale

        feature_size = features.shape[2:]

        boxes_h[:, 1:] = clip_boxes_to_image(boxes_h[:, 1:], feature_size)
        boxes_o[:, 1:] = clip_boxes_to_image(boxes_o[:, 1:], feature_size)

        for idx, mask in enumerate(masks):
            mask_h = self.fill_masks_for_boxes(mask, boxes_h[idx, 1:])
            mask_o = self.fill_masks_for_boxes(mask, boxes_o[idx, 1:])
            masks[idx] = torch.max(mask_h, mask_o)

        return masks

    def compute_box_pair_union(self, boxes_1, boxes_2):
        """
        Compute box pair union given box coordinates with batch index

        Arguments:
            boxes_1(Tensor[N, 5])
            boxes_2(Tensor[N, 5])
        """
        boxes_1[:, 1] = torch.min(boxes_1[:, 1], boxes_2[:, 1])
        boxes_1[:, 2] = torch.min(boxes_1[:, 2], boxes_2[:, 2])
        boxes_1[:, 3] = torch.max(boxes_1[:, 3], boxes_2[:, 3])
        boxes_1[:, 4] = torch.max(boxes_1[:, 4], boxes_2[:, 4])
        
        return boxes_1

    def forward(self, features, boxes_h, boxes_o):
        """
        Arguments:
            features(list[Tensor[N, C, H, W]]): Feature pyramid, with each element 
                representating a particular level
            boxes_h(list[Tensor[M, 4]]) 
            boxes_o(list[Tensor[M, 4]])
        """
        assert len(features) == self.num_levels,\
            "Number of levels in given features does not match the number of scales"
 
        boxes_h = convert_boxes_to_roi_format(boxes_h)
        boxes_o = convert_boxes_to_roi_format(boxes_o)

        num_boxes = len(boxes_h)
        num_iter = num_boxes // self.mask_limit + bool(num_boxes % self.mask_limit)
        output = torch.zeros(num_boxes, features[0].shape[1], *self.output_size,
            dtype=features[0].dtype,
            device=features[0].device,
        )
        # Compute pooled features iteratively based on maximum number of masks allowed
        for idx in range(num_iter):
            start_idx = idx * self.mask_limit
            end_idx = min(start_idx + self.mask_limit, num_boxes)

            box_pair_union = self.compute_box_pair_union(
                boxes_h[start_idx: end_idx],
                boxes_o[start_idx: end_idx]
            )

            if self.num_levels == 1:
                box_pair_masks = self.construct_masks_for_box_pairs(
                    features[0], 0,
                    boxes_h[start_idx: end_idx],
                    boxes_o[start_idx: end_idx]
                )
                output[start_idx: end_idx] = \
                    masked_roi_align(
                        features[0],
                        box_pair_union,
                        box_pair_masks,
                        self.output_size,
                        spatial_scale=self.spatial_scale[0],
                        sampling_ratio=self.sampling_ratio,
                        clone_limit=self.mask_limit
                    )
            else:
                levels = self.map_levels(
                    boxes_h[start_idx: end_idx],
                    boxes_o[start_idx: end_idx]
                )
                for level, (per_level_feature, scale) in enumerate(zip(features, self.spatial_scale)):
                    idx_in_level = torch.nonzero(levels == level).squeeze(1)
                    rois_per_level = box_pair_union[idx_in_level]
                    masks_per_level = self.construct_masks_for_box_pairs(
                        per_level_feature, level,
                        boxes_h[start_idx: end_idx][idx_in_level],
                        boxes_o[start_idx: end_idx][idx_in_level]
                    )

                    output[start_idx: end_idx][idx_in_level] = \
                        masked_roi_align(
                            per_level_feature,
                            rois_per_level,
                            masks_per_level,
                            self.output_size,
                            spatial_scale=scale,
                            sampling_ratio=self.sampling_ratio,
                            clone_limit=self.mask_limit
                        )

        return output

