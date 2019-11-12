"""
Faster R-CNN with different backbones based on
torchvision implementation

Written by Frederic Zhang
Australian National University

Last updated in Jun. 2019
"""

from collections import OrderedDict

from torch import nn
from torchvision import models
from torchvision.models.detection import FasterRCNN

class BackboneForFasterRCNN(nn.ModuleDict):
    """
    Prepare backbone architecture for Faster R-CNN

    Keep the backbone network up to a specified layer, assuming layers in the
    backbone are registered in the same order as execution in forward pass

    Append number of output channels of the backbone network as a class attribute,
    as per requirement of the FasterRCNN module
    
    Arguments:
        backbone(Module): Backbone architecture, typically with FC layers
        return_layer(str): Name of layer to take output from
        out_channels(int): Number of output channels of the backbone network
    """
    def __init__(self, backbone, return_layer, out_channels):
        layers = OrderedDict()
        for name, module in backbone.named_children():
            layers[name] = module
            if name == return_layer:
                break
        super(BackboneForFasterRCNN, self).__init__(layers)
        self.out_channels = out_channels

    def forward(self, x):
        for name, module in self.named_children():
            x = module(x)
        return x

_models = {
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152
        }

def fasterrcnn_resnet(backbone_name,
        num_classes=91, pretrained_backbone=True, **kwargs):
    """
    Construct a Faster R-CNN network, using a specified backbone

    Arguments:
        backbone_name(str): Name of the backbone
        num_classes(int, optional): Number of target classes, default: 91(COCO)
        pretrained_backbone(bool, optional): Use the backbone with weights trained on ImageNet

        Refer to torchvision.models.detection.FasterRCNN for kwargs
    """

    model = _models[backbone_name](
            pretrained=pretrained_backbone)
    backbone = BackboneForFasterRCNN(model, 'layer4', 2048)

    model = FasterRCNN(backbone, num_classes, **kwargs)

    return model
