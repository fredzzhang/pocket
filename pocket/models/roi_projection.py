"""
RoI projection utilities

Written by Frederic Zhang
Australian National University

Last updated in Aug. 2019
"""

import torch

from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class RoIFeatureExtractor(nn.Module):
    """
    RoI feature extractor using Faster R-CNN with ResNet50-FPN

    The features are extracted from fc7 as illustrated below
        ...
        |- Residual block 5 (NxNx2048)
        |- RoI pooling (7x7x2048)
        |- fc6 (1024)
        |- fc7 (1024)
    """
    def __init__(self):
        super(RoIFeatureExtractor, self).__init__()
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.detector.eval()

    def forward(self, images, boxes):
        """
        Extract RoI features

        Arguments:
            images(list[Tensor]): Images to be processed
            boxes(list[Tensor[N, 4]]): Bounding boxes arranged in format (x1, y1, x2, y2)
        """
        # Modified from torchvision.models.detection.GeneralizedRCNN
        # and torchvision.models.detection.RoIHeads
        # Copyright (c) Facebook, Inc. and its affiliates
        original_image_sizes = [img.shape[-2:] for img in images]
        images, _ = self.detector.transform(images)
        # resize the bounding boxes
        for i, (h, w) in enumerate(images.image_sizes):
            scale_h = float(h) / original_image_sizes[i][0]
            scale_w = float(w) / original_image_sizes[i][1]
            assert abs(scale_h - scale_w) < 1e-2,\
                    'Unequal scaling factor'
            boxes[i] *= (scale_h + scale_w) / 2
        features = self.detector.backbone(images.tensors)
        box_features = self.detector.roi_heads.box_roi_pool(
                features,
                boxes,
                images.image_sizes)
        box_features = self.detector.roi_heads.box_head(box_features)

        return box_features

class RoIProjector(RoIFeatureExtractor):
    """
    Project RoIs onto an image
    """
    def __init__(self):
        super(RoIProjector, self).__init__()

    def forward(self, images, boxes):
        """
        Compute the feature representation and class logits for given RoIs

        Arguments:
            images(list[Tensor]): Images to be processed
            boxes(list[Tensor[N, 4]]): Bounding boxes arranged in format (x1, y1, x2, y2)
        """
        box_features = super(RoIProjector, self).forward(images, boxes)
        class_logits, _ = self.detector.roi_heads.box_predictor(box_features)

        return box_features, class_logits

