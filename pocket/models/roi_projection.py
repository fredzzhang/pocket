"""
Utilities related to RoI projection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from torch.nn.functional import relu
from .faster_rcnn import fasterrcnn_resnet_fpn

class RoIFeatureExtractor:
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
        backbone_name(str, optional):
        pretrained(bool, optional):
    """
    def __init__(self, return_layer='fc7', backbone_name='resnet50', pretrained=True):
        self._return_layer = return_layer
        self._backbone_name = backbone_name
        self._pretrained = pretrained

        self.detector = fasterrcnn_resnet_fpn(backbone_name, pretrained)
        self.detector.eval()

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += 'return_layer=\"'
        reprstr += self._return_layer
        reprstr += '\", backbone_name=\"'
        reprstr += self._backbone_name
        reprstr += '\", pretrained='
        reprstr += str(self._pretrained)
        reprstr += ')'
        return reprstr

    def __call__(self, images, boxes):
        """
        Extract RoI features

        Arguments:
            images(list[Tensor]): Images to be processed
            boxes(list[Tensor[N, 4]]): Bounding boxes arranged in format (x1, y1, x2, y2)

        Returns:
            Tensor[M, ...]: Features corresponding to different images are stacked in order
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

        if self._return_layer == 'roi_pool':
            return box_features
        elif self._return_layer == 'fc6':
            box_features = box_features.flatten(start_dim=1)
            return self.detector.roi_heads.box_head.fc6(box_features)
        elif self._return_layer == 'fc7':
            box_features = box_features.flatten(start_dim=1)
            box_features = relu(self.detector.roi_heads.box_head.fc6(box_features))
            return self.detector.roi_heads.box_head.fc7(box_features)

class RoIProjector(RoIFeatureExtractor):
    """
    Project RoIs onto an image
    """
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super().__init__('fc7', backbone_name, pretrained)

    def __call__(self, images, boxes):
        """
        Compute the feature representation and class logits for given RoIs

        Arguments:
            images(list[Tensor]): Images to be processed
            boxes(list[Tensor[N, 4]]): Bounding boxes arranged in format (x1, y1, x2, y2)

        Returns:
            Tensor[M, 1024]: fc7 features stacked in order
            Tensor[M, 91]: Predicted scores for each class including background
        """
        box_features = super().__call__(images, boxes)
        class_logits, _ = self.detector.roi_heads.box_predictor(relu(box_features))

        return box_features, class_logits

