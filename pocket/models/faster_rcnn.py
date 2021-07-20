"""
Faster R-CNN with different backbones based on
torchvision implementation

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

from collections import OrderedDict

import torch
import warnings
from torch import nn
from torchvision import models
from torchvision.ops import misc as misc_nn_ops
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

__all__ = [
    'fasterrcnn_resnet',
    'fasterrcnn_resnet_fpn',
    'fasterrcnn_resnet_fpn_x'
]

def resnet_backbone(backbone_name, pretrained):
    backbone = models.resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    # ResNet C4 features are used as per Mask R-CNN paper
    return IntermediateLayerGetter(backbone, {'layer3': 0}), backbone.layer4

def fasterrcnn_resnet(backbone_name,
        num_classes=91, pretrained_backbone=True, **kwargs):
    """
    Construct Faster R-CNN with a ResNet backbone

    Arguments:
        backbone_name(str): Name of the backbone.
            Refer to torchvision.models.resnet.__dict__ for details
        num_classes(int, optional): Number of target classes, default: 91(COCO)
        pretrained_backbone(bool, optional): If True, load weights for backbone
            pre-trained on ImageNet

        Refer to torchvision.models.detection.FasterRCNN for kwargs
    """

    backbone, res5 = resnet_backbone(backbone_name, pretrained_backbone)
    backbone.out_channels = 1024
    box_head = nn.Sequential(
        res5,
        nn.AdaptiveAvgPool2d((1, 1))
    )
    box_predictor = FastRCNNPredictor(2048, num_classes)
    model = FasterRCNN(backbone,
        box_head=box_head, box_predictor=box_predictor, **kwargs)

    return model

model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}

KEEP = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
    88, 89, 90
]
KEEPX4 = torch.arange(4).repeat(81, 1) + torch.as_tensor(KEEP).unsqueeze(1) * 4

def _validate_trainable_layers(pretrained, trainable_backbone_layers, max_value, default_value):
    # dont freeze any layers if pretrained model or backbone is not used
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                "falling back to trainable_backbone_layers={} so that all layers are trainable".format(max_value))
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    assert 0 <= trainable_backbone_layers <= max_value
    return trainable_backbone_layers

def fasterrcnn_resnet_fpn(backbone_name, pretrained=False, trainable_backbone_layers=None,
        num_classes=81, pretrained_backbone=True, **kwargs):
    """
    Construct Faster R-CNN with a ResNet-FPN backbone

    Arguments:
        backbone_name(str): Name of the backbone.
            Refer to torchvision.models.resnet.__dict__ for details
        pretrained(bool, optional): If True, load weights for the detector
            pretrained on MS COCO. Only ResNet50-FPN is supported for the moment.
        trainable_backbone_layers(int, optional): Number of trainable (not frozen)
            resnet layers starting from final block.
        num_classes(int, optional): Number of target classes.
        pretrained_backbone(bool, optional): If True, load weights for backbone
            pre-trained on ImageNet

        Refer to torchvision.models.detection.FasterRCNN for kwargs
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained and backbone_name == 'resnet50':
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone(backbone_name, pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained and backbone_name == 'resnet50':
        state_dict = models.utils.load_state_dict_from_url(
            model_urls['fasterrcnn_resnet50_fpn_coco'])
        if num_classes == 81:
            # Remove the parameters for the additional classes
            state_dict['roi_heads.box_predictor.cls_score.weight'] = \
                state_dict['roi_heads.box_predictor.cls_score.weight'][KEEP]
            state_dict['roi_heads.box_predictor.cls_score.bias'] = \
                state_dict['roi_heads.box_predictor.cls_score.bias'][KEEP]
            state_dict['roi_heads.box_predictor.bbox_pred.weight'] = \
                state_dict['roi_heads.box_predictor.bbox_pred.weight'][KEEPX4.flatten()]
            state_dict['roi_heads.box_predictor.bbox_pred.bias'] = \
                state_dict['roi_heads.box_predictor.bbox_pred.bias'][KEEPX4.flatten()]

        model.load_state_dict(state_dict)
    elif pretrained:
        print("WARNING: No pretrained detector on MS COCO with {}.".format(backbone_name),
            "Proceed with only pretrained backbone on ImageNet.")
    return model

class FasterRCNN_(nn.Module):
    """
    Modified Faster R-CNN

    By default, the RoI head performs regression in conjunction with classification.
    This introduces a mismatch between box scores and the regressed coordinates. Also,
    class-specific regression results in a single score kept for each box, as opposed
    to a probability distribution over target classes. This module pools the already
    regressed boxes and addds an extra round of classification to give a more reliable
    score distribution for all object classes

    Arguments:
        frccn(GeneralizedRCNN): An instantiated Faster R-CNN model
    """
    def __init__(self, frcnn):
        super().__init__()
        self.transform = frcnn.transform
        self.backbone = frcnn.backbone
        self.rpn = frcnn.rpn
        self.roi_heads = frcnn.roi_heads

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            dict[Tensor]: During training, return a dict that contains the losses
            list[dict]: During testing, return dicts of detected boxes
                "boxes": Tensor[M, 4]
                "scores": Tensor[M, C]
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # Extract the regressed boxes
        detections = [det['boxes'] for det in detections]
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in detections]
        # RoI reprojection
        box_features = self.roi_heads.box_roi_pool(features, detections, images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        # Classification
        class_logits, _ = self.roi_heads.box_predictor(box_features)
        pred_scores = nn.functional.softmax(class_logits, -1).split(boxes_per_image, 0)
        # Format the detections
        detections = [
            dict(boxes=boxes_in_image, scores=scores_in_image) for
                boxes_in_image, scores_in_image in zip(detections, pred_scores)
        ]

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections

def fasterrcnn_resnet_fpn_x(*args, **kwargs):
    """Instantiate FRCNN-ResNet-FPN with extra RoI projection"""
    return FasterRCNN_(fasterrcnn_resnet_fpn(*args, **kwargs))
