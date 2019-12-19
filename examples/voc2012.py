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
        batch_size=32, shuffle=True, num_workers=4
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
    engine(5)

if __name__ == '__main__':
    main()

    # Sample output
    """
    => Validation (+64.57s)
    Epoch: 0 | mAP: 0.0888 | Loss: 6.4674 | Time: 55.74s

    [Ep.][Iter.]: [1][50] | Loss: 0.3516 | Time[Data/Iter.]: [0.8834s/44.9455s]
    [Ep.][Iter.]: [1][100] | Loss: 0.2623 | Time[Data/Iter.]: [0.0115s/32.8341s]
    [Ep.][Iter.]: [1][150] | Loss: 0.2550 | Time[Data/Iter.]: [0.0088s/33.2330s]

    => Training (+211.59s)
    Epoch: 1 | mAP: 0.0929 | Time(eval): 2.91s
    => Validation (+254.78s)
    Epoch: 1 | mAP: 0.1319 | Loss: 0.3520 | Time: 43.20s

    [Ep.][Iter.]: [2][200] | Loss: 0.2435 | Time[Data/Iter.]: [0.9589s/45.9702s]
    [Ep.][Iter.]: [2][250] | Loss: 0.2466 | Time[Data/Iter.]: [0.0128s/33.3668s]
    [Ep.][Iter.]: [2][300] | Loss: 0.2366 | Time[Data/Iter.]: [0.0150s/33.2839s]
    [Ep.][Iter.]: [2][350] | Loss: 0.2354 | Time[Data/Iter.]: [0.0145s/33.4185s]

    => Training (+378.84s)
    Epoch: 2 | mAP: 0.1353 | Time(eval): 2.62s
    => Validation (+423.25s)
    Epoch: 2 | mAP: 0.1508 | Loss: 0.2576 | Time: 44.40s

    [Ep.][Iter.]: [3][400] | Loss: 0.2307 | Time[Data/Iter.]: [0.9710s/33.3509s]
    [Ep.][Iter.]: [3][450] | Loss: 0.2307 | Time[Data/Iter.]: [0.0120s/32.9732s]
    [Ep.][Iter.]: [3][500] | Loss: 0.2267 | Time[Data/Iter.]: [0.0110s/32.9218s]

    => Training (+545.57s)
    Epoch: 3 | mAP: 0.1700 | Time(eval): 2.59s
    => Validation (+588.90s)
    Epoch: 3 | mAP: 0.1838 | Loss: 0.2866 | Time: 43.33s

    [Ep.][Iter.]: [4][550] | Loss: 0.2278 | Time[Data/Iter.]: [0.9267s/33.3011s]
    [Ep.][Iter.]: [4][600] | Loss: 0.2225 | Time[Data/Iter.]: [0.0127s/32.7028s]
    [Ep.][Iter.]: [4][650] | Loss: 0.2260 | Time[Data/Iter.]: [0.0116s/32.9071s]
    [Ep.][Iter.]: [4][700] | Loss: 0.2214 | Time[Data/Iter.]: [0.0117s/32.8610s]

    => Training (+711.95s)
    Epoch: 4 | mAP: 0.1878 | Time(eval): 2.58s
    => Validation (+755.31s)
    Epoch: 4 | mAP: 0.1820 | Loss: 0.2649 | Time: 43.36s

    [Ep.][Iter.]: [5][750] | Loss: 0.2237 | Time[Data/Iter.]: [0.5999s/33.0283s]
    [Ep.][Iter.]: [5][800] | Loss: 0.2171 | Time[Data/Iter.]: [0.0119s/33.0726s]
    [Ep.][Iter.]: [5][850] | Loss: 0.2191 | Time[Data/Iter.]: [0.0113s/32.9399s]

    => Training (+878.36s)
    Epoch: 5 | mAP: 0.2033 | Time(eval): 2.59s
    => Validation (+921.95s)
    Epoch: 5 | mAP: 0.2207 | Loss: 0.2388 | Time: 43.59s
    """
