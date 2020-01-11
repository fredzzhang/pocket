"""
Multilabel classification example on HICO-DET

The code employs Faster R-CNN (ResNet-50-FPN)
pretrained on MS COCO as a feature extractor.
Features (fc7) for the union of ground truth
box pairs are computed and fed into a simple
MLP to compute class logits for the 600 interactions.

This method has abysmal performance (2% mAP)
and only serves as an example to demonstrate
the usage of pocket.core.MultiLabelClassificationEngine

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.ops.boxes import box_iou

from pocket.data import HICODet
from pocket.models import RoIFeatureExtractor, MultiLayerPerceptron
from pocket.core import MultiLabelClassificationEngine

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = RoIFeatureExtractor()
        self.head = MultiLayerPerceptron([1024, 1024, 600], [True, True])

    def forward(self, image, boxes):
        with torch.no_grad():
            f = self.ext(image, boxes)
        return self.head(f)

    """Override methods to focus on classification head"""
    def parameters(self):
        return self.head.parameters()
    def state_dict(self):
        return self.head.state_dict()
    def load_state_dict(self, state_dict):
        self.head.load_state_dict(state_dict)

def transforms(image, target):
    """Transform image and target to desired format for learning engine"""
    image = to_tensor(image)

    boxes_h = torch.as_tensor(target['boxes_h'])
    boxes_o = torch.as_tensor(target['boxes_o'])
    boxes = torch.zeros_like(boxes_h)
    boxes[:, :2] = torch.min(boxes_h[:, :2], boxes_o[:, :2])
    boxes[:, 2:] = torch.max(boxes_h[:, 2:], boxes_o[:, 2:])

    hoi = torch.as_tensor(target['hoi'])
    labels = torch.zeros(len(hoi), 600)

    # Associate ground truth box pairs that have IoU higher than 0.5
    min_iou = torch.min(box_iou(boxes_h, boxes_h), box_iou(boxes_o, boxes_o))
    match = torch.nonzero(min_iou > 0.5)
    labels[match[:, 0], hoi[match[:, 1]]] = 1

    return image, boxes, labels

def custom_collate(batch):
    image = []
    boxes = []
    labels = []
    for instance in batch:
        image.append(instance[0])
        boxes.append(instance[1])
        labels.append(instance[2])
    # Combine inputs into a list while stacking labels in a tensor
    return image, boxes, torch.cat(labels, 0)

if __name__ == '__main__':

    HICO_ROOT = "./data/hicodet"
    if not os.path.exists(HICO_ROOT):
        raise ValueError("Cannot find the dataset"
            "Make sure a symbolic link is created at {}".format(HICO_ROOT))

    net = Net()

    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        HICODet(
            root=(HICO_ROOT, "hico_20160224_det/images/train2015"),
            annoFile=os.path.join(HICO_ROOT, "instances_train2015.json"),
            transforms=transforms
        ), batch_size=4, shuffle=True, num_workers=4,
        collate_fn=custom_collate, drop_last=True
    )
    test_loader = DataLoader(
        HICODet(
            root=os.path.join(HICO_ROOT, "hico_20160224_det/images/test2015"),
            annoFile=os.path.join(HICO_ROOT, "instances_test2015.json"),
            transforms=transforms
        ), batch_size=4, num_workers=4,
        collate_fn=custom_collate, drop_last=True
    )

    engine = MultiLabelClassificationEngine(net, criterion, train_loader,
        val_loader=test_loader, ap_algorithm='11P', print_interval=500)

    engine(5)
