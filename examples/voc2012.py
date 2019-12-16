"""
An example of multilabel classification on voc2012

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
from torchvision import datasets, models, transforms
from pocket.utils import MultiLabelClassificationEngine

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
    => Validation (+57.05s)
    Epoch: 0 | mAP: 0.0888 | Loss: 6.4674 | Time: 54.01s

    [Ep.][Iter.]: [1][50] | Loss: 0.3601 | Time[Data][Iter.]: [26.5870s][0.9164s]
    [Ep.][Iter.]: [1][100] | Loss: 0.2634 | Time[Data][Iter.]: [19.1039s][0.0111s]
    [Ep.][Iter.]: [1][150] | Loss: 0.2532 | Time[Data][Iter.]: [19.2337s][0.0104s]

    => Training (+195.74s)
    Epoch: 1 | mAP: 0.0925 | Time(eval): 2.35s
    => Validation (+238.64s)
    Epoch: 1 | mAP: 0.1283 | Loss: 0.4617 | Time: 42.90s

    [Ep.][Iter.]: [2][200] | Loss: 0.2417 | Time[Data][Iter.]: [30.7052s][0.8373s]
    [Ep.][Iter.]: [2][250] | Loss: 0.2450 | Time[Data][Iter.]: [19.2139s][0.0101s]
    [Ep.][Iter.]: [2][300] | Loss: 0.2337 | Time[Data][Iter.]: [19.2366s][0.0103s]
    [Ep.][Iter.]: [2][350] | Loss: 0.2342 | Time[Data][Iter.]: [19.1734s][0.0101s]

    => Training (+360.82s)
    Epoch: 2 | mAP: 0.1395 | Time(eval): 2.20s
    => Validation (+404.11s)
    Epoch: 2 | mAP: 0.1502 | Loss: 0.2780 | Time: 43.29s

    [Ep.][Iter.]: [3][400] | Loss: 0.2314 | Time[Data][Iter.]: [18.9970s][1.0269s]
    [Ep.][Iter.]: [3][450] | Loss: 0.2296 | Time[Data][Iter.]: [19.1553s][0.0154s]
    [Ep.][Iter.]: [3][500] | Loss: 0.2265 | Time[Data][Iter.]: [19.0785s][0.0121s]

    => Training (+525.67s)
    Epoch: 3 | mAP: 0.1695 | Time(eval): 2.36s
    => Validation (+568.71s)
    Epoch: 3 | mAP: 0.1742 | Loss: 0.2734 | Time: 43.04s

    [Ep.][Iter.]: [4][550] | Loss: 0.2273 | Time[Data][Iter.]: [19.0480s][0.8987s]
    [Ep.][Iter.]: [4][600] | Loss: 0.2213 | Time[Data][Iter.]: [19.2186s][0.0124s]
    [Ep.][Iter.]: [4][650] | Loss: 0.2243 | Time[Data][Iter.]: [19.3367s][0.0135s]
    [Ep.][Iter.]: [4][700] | Loss: 0.2223 | Time[Data][Iter.]: [19.2185s][0.0146s]

    => Training (+690.65s)
    Epoch: 4 | mAP: 0.1898 | Time(eval): 2.47s
    => Validation (+734.37s)
    Epoch: 4 | mAP: 0.1841 | Loss: 0.2249 | Time: 43.72s

    [Ep.][Iter.]: [5][750] | Loss: 0.2218 | Time[Data][Iter.]: [18.9964s][1.0675s]
    [Ep.][Iter.]: [5][800] | Loss: 0.2164 | Time[Data][Iter.]: [19.2012s][0.0105s]
    [Ep.][Iter.]: [5][850] | Loss: 0.2185 | Time[Data][Iter.]: [19.2428s][0.0108s]

    => Training (+856.15s)
    Epoch: 5 | mAP: 0.2091 | Time(eval): 2.36s
    => Validation (+899.38s)
    Epoch: 5 | mAP: 0.2131 | Loss: 0.2384 | Time: 43.23s
    """
