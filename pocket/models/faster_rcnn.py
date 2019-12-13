"""
Faster R-CNN with different backbones based on
torchvision implementation

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from torch import nn
from torchvision import models
from torchvision.ops import misc as misc_nn_ops
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

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

def fasterrcnn_resnet_fpn(backbone_name, pretrained=False,
        num_classes=91, pretrained_backbone=True, **kwargs):
    """
    Construct Faster R-CNN with a ResNet-FPN backbone

    Arguments:
        backbone_name(str): Name of the backbone.
            Refer to torchvision.models.resnet.__dict__ for details
        pretrained(bool, optional): If True, load weights for the detector
            pretrained on COCO. Only ResNet50-FPN is supported for the moment.
        num_classes(int, optional): Number of target classes, default: 91(COCO)
        pretrained_backbone(bool, optional): If True, load weights for backbone
            pre-trained on ImageNet

        Refer to torchvision.models.detection.FasterRCNN for kwargs
    """
    if pretrained and backbone_name == 'resnet50':
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone(backbone_name, pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained and backbone_name == 'resnet50':
        state_dict = models.utils.load_state_dict_from_url(
            model_urls['fasterrcnn_resnet50_fpn_coco'])
        model.load_state_dict(state_dict)
    elif pretrained:
        print("WARNING: No pretrained detector for backbone {}.".format(backbone_name),
            "Proceed with only pretrained backbone.")
    return model