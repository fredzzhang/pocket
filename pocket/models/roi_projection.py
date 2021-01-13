"""
Utilities related to RoI projection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

"""
Acknowledgement:

Source code in this module is largely modified from
https://github.com/pytorch/vision/tree/master/torchvision/models/detection

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

from torch import nn
from torch.nn.functional import relu
from .faster_rcnn import fasterrcnn_resnet_fpn

class RoIFeatureExtractor(nn.Module):
    """
    RoI feature extractor using Faster R-CNN with ResNet-FPN

    The features are extracted from fc7 as illustrated below
        ...
        |- c2 (NxNx256)
        |- c3 (NxNx512)
        |- c4 (NxNx1024)
        |- c5 (NxNx2048)
        |- roi_pool (7x7x256)
        |- fc6 (1024)
        |- fc7 (1024)

    Arguments:
        return_layer(str, optional): The specific layer to extract feature from.
            A choice amongst 'roi_pool', 'fc6' and 'fc7'
        backbone_name(str, optional): Name of the backbone.
            Refer to torchvision.models.resnet.__dict__ for details
        pretrained(bool, optional): If True, use pretrained weights on COCO

    Example:

        >>> import torch
        >>> from pocket.models import RoIFeatureExtractor()
        >>> m = RoIFeatureExtractor()
        >>> image = torch.rand(3, 512, 512)
        >>> boxes = torch.rand(5, 4) * 256; boxes[:, 2:] += boxes[:, :2]
        >>> f = m([image], [boxes])
    """
    def __init__(self, return_layer='fc7', backbone_name='resnet50', pretrained=True):
        super().__init__()
        self._return_layer = return_layer
        self._backbone_name = backbone_name
        self._pretrained = pretrained

        detector = fasterrcnn_resnet_fpn(backbone_name, pretrained)

        self.transform = detector.transform
        self.backbone = detector.backbone

        self.roi_pool = detector.roi_heads.box_roi_pool
        if return_layer == 'pool':
            self.roi_heads = None
        elif return_layer == 'fc6':
            self.roi_heads = detector.roi_heads.fc6
        elif return_layer == 'fc7':
            self.roi_heads = nn.Sequential(
                detector.roi_heads.box_head.fc6,
                nn.ReLU(),
                detector.roi_heads.box_head.fc7
            )
        else:
            raise ValueError("Specified return layer does not exist")

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += 'return_layer='
        reprstr += repr(self._return_layer)
        reprstr += ', backbone_name='
        reprstr += repr(self._backbone_name)
        reprstr += ', pretrained='
        reprstr += str(self._pretrained)
        reprstr += ')'
        return reprstr

    def forward(self, images, boxes):
        """
        Extract RoI features

        Arguments:
            images(list[Tensor]): Images to be processed
            boxes(list[Tensor[N, 4]]): Bounding boxes arranged in format (x1, y1, x2, y2)

        Returns:
            Tensor[M, ...]: Features corresponding to different images are stacked in order
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        images, _ = self.transform(images)
        # resize the bounding boxes
        for i, (h, w) in enumerate(images.image_sizes):
            scale_h = float(h) / original_image_sizes[i][0]
            scale_w = float(w) / original_image_sizes[i][1]
            assert abs(scale_h - scale_w) < 1e-2,\
                    'Unequal scaling factor'
            boxes[i] *= (scale_h + scale_w) / 2
        features = self.backbone(images.tensors)
        box_features = self.roi_pool(
            features,
            boxes,
            images.image_sizes
        )

        if self._return_layer == 'pool':
            return box_features
        else:
            box_features = box_features.flatten(start_dim=1)
            return self.roi_heads(box_features)

class RoIProjector(nn.Module):
    """
    Project RoIs onto an image
    """
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super().__init__()
        self._backbone_name = backbone_name
        self._pretrained = pretrained

        detector = fasterrcnn_resnet_fpn(backbone_name, pretrained)

        self.transform = detector.transform
        self.backbone = detector.backbone
        self.roi_heads = detector.roi_heads

    def forward(self, images, boxes):
        """
        Compute the feature representation and class logits for given RoIs

        Arguments:
            images(list[Tensor]): Images to be processed
            boxes(list[Tensor[N, 4]]): Bounding boxes arranged in format (x1, y1, x2, y2)

        Returns:
            Tensor[M, 1024]: fc7 features stacked in order
            Tensor[M, 91]: Predicted scores for each class including background
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        images, _ = self.transform(images)
        # resize the bounding boxes
        for i, (h, w) in enumerate(images.image_sizes):
            scale_h = float(h) / original_image_sizes[i][0]
            scale_w = float(w) / original_image_sizes[i][1]
            assert abs(scale_h - scale_w) < 1e-2,\
                    'Unequal scaling factor'
            boxes[i] *= (scale_h + scale_w) / 2
        features = self.backbone(images.tensors)

        box_features = self.roi_heads.box_roi_pool(
            features,
            boxes,
            images.image_sizes
        )
        box_features = self.roi_heads.box_head(box_features)

        class_logits, _ = self.roi_heads.box_predictor(box_features)
        pred_scores = nn.functional.softmax(class_logits, -1)

        return box_features, pred_scores
