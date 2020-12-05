"""
An example of multilabel classification on voc2012

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
from torchvision import datasets, models, transforms
from pocket.core import MultiLabelClassificationEngine

CLASSES = (
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor",
    )
NUM_CLASSES = len(CLASSES)

def target_transform(x):
    target = torch.zeros(NUM_CLASSES)
    anno = x['annotation']['object']
    if isinstance(anno, list):
        for obj in anno:
            target[CLASSES.index(obj['name'])] = 1
    else:
        target[CLASSES.index(anno['name'])] = 1
    return target

def main():
    # Fix random seed
    torch.manual_seed(0)
    # Initialize network
    net = models.resnet50(num_classes=NUM_CLASSES)
    # Initialize loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # Prepare dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.VOCDetection('./data', image_set='train', download=True,
            transform=transforms.Compose([
                transforms.Resize([480, 480]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            target_transform=target_transform),
        batch_size=16, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.VOCDetection('./data', image_set='val',
            transform=transforms.Compose([
                transforms.Resize([480, 480]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            target_transform=target_transform),
        batch_size=32, num_workers=4
    )
    # Initialize learning engine and start training
    engine = MultiLabelClassificationEngine(net, criterion, train_loader,
        val_loader=val_loader, print_interval=50,
        optim_params={
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4
        })
    # Train the network for one epoch with default optimizer option
    # Checkpoints will be saved under ./checkpoints by default, containing 
    # saved model parameters, optimizer statistics and progress
    engine(1)

if __name__ == '__main__':
    main()

    # Sample output
    """
    => Validation (+66.04s)
    Epoch: 0 | mAP: 0.0886 | Loss: 6.4674 | Time: 64.35s

    Epoch [1/1], Iter. [050/358], Loss: 0.3914, Time[Data/Iter.]: [0.97s/29.54s]
    Epoch [1/1], Iter. [100/358], Loss: 0.2695, Time[Data/Iter.]: [0.63s/29.91s]
    Epoch [1/1], Iter. [150/358], Loss: 0.2507, Time[Data/Iter.]: [0.63s/30.26s]
    Epoch [1/1], Iter. [200/358], Loss: 0.2489, Time[Data/Iter.]: [0.63s/30.12s]
    Epoch [1/1], Iter. [250/358], Loss: 0.2463, Time[Data/Iter.]: [0.63s/30.05s]
    Epoch [1/1], Iter. [300/358], Loss: 0.2478, Time[Data/Iter.]: [0.64s/30.00s]
    Epoch [1/1], Iter. [350/358], Loss: 0.2396, Time[Data/Iter.]: [0.65s/30.00s]

    => Training (+282.08s)
    Epoch: 1 | mAP: 0.1004 | Time(eval): 0.85s
    => Validation (+347.15s)
    Epoch: 1 | mAP: 0.1433 | Loss: 0.2515 | Time: 65.07s

    """
