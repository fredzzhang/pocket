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
            root=os.path.join(HICO_ROOT, "hico_20160224_det/images/train2015"),
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
        collate_fn=custom_collate
    )

    engine = MultiLabelClassificationEngine(net, criterion, train_loader,
        val_loader=test_loader, ap_algorithm='11P', print_interval=500)

    engine(5)


    """
    Sample output


    => Validation (+1754.71s)
    Epoch: 0 | mAP: 0.0067 | Loss: 0.6940 | Time: 1751.42s
    
    ...
    ...
    ...

    [Ep.][Iter.]: [5][38000] | Loss: 0.0388 | Time[Data/Iter.]: [0.4588s/203.3094s]
    [Ep.][Iter.]: [5][38500] | Loss: 0.0383 | Time[Data/Iter.]: [0.0629s/206.3993s]
    [Ep.][Iter.]: [5][39000] | Loss: 0.0387 | Time[Data/Iter.]: [0.0600s/196.8182s]
    [Ep.][Iter.]: [5][39500] | Loss: 0.0387 | Time[Data/Iter.]: [0.0607s/208.0784s]
    [Ep.][Iter.]: [5][40000] | Loss: 0.0383 | Time[Data/Iter.]: [0.0621s/200.6665s]
    [Ep.][Iter.]: [5][40500] | Loss: 0.0382 | Time[Data/Iter.]: [0.0621s/198.3799s]
    [Ep.][Iter.]: [5][41000] | Loss: 0.0376 | Time[Data/Iter.]: [0.0643s/206.2081s]
    [Ep.][Iter.]: [5][41500] | Loss: 0.0378 | Time[Data/Iter.]: [0.0619s/197.4531s]
    [Ep.][Iter.]: [5][42000] | Loss: 0.0370 | Time[Data/Iter.]: [0.0660s/199.5019s]
    [Ep.][Iter.]: [5][42500] | Loss: 0.0364 | Time[Data/Iter.]: [0.0620s/202.1312s]
    [Ep.][Iter.]: [5][43000] | Loss: 0.0370 | Time[Data/Iter.]: [0.0622s/199.2598s]
    [Ep.][Iter.]: [5][43500] | Loss: 0.0369 | Time[Data/Iter.]: [0.0640s/203.9033s]
    [Ep.][Iter.]: [5][44000] | Loss: 0.0367 | Time[Data/Iter.]: [0.0625s/200.8627s]
    [Ep.][Iter.]: [5][44500] | Loss: 0.0358 | Time[Data/Iter.]: [0.0637s/198.3094s]
    [Ep.][Iter.]: [5][45000] | Loss: 0.0358 | Time[Data/Iter.]: [0.0645s/198.1471s]
    [Ep.][Iter.]: [5][45500] | Loss: 0.0365 | Time[Data/Iter.]: [0.0633s/206.5252s]
    [Ep.][Iter.]: [5][46000] | Loss: 0.0362 | Time[Data/Iter.]: [0.0661s/205.7809s]
    [Ep.][Iter.]: [5][46500] | Loss: 0.0354 | Time[Data/Iter.]: [0.0639s/200.6037s]
    [Ep.][Iter.]: [5][47000] | Loss: 0.0355 | Time[Data/Iter.]: [0.0596s/201.4554s]

    => Training (+25678.19s)
    Epoch: 5 | mAP: 0.0114 | Time(eval): 100.55s
    => Validation (+26641.95s)
    Epoch: 5 | mAP: 0.0255 | Loss: 0.0284 | Time: 963.73s
    """
