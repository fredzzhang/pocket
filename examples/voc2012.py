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
        assert isinstance(anno, dict)
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
    => Validation
    Epoch: 0 | mAP: 0.0888 | Loss: 6.4674 | Time: 95.99s

    [Ep.][Iter.]: [1][50] | Loss: 0.3523 | Time[Data][Iter.]: [56.7899s][0.7396s]
    [Ep.][Iter.]: [1][100] | Loss: 0.2589 | Time[Data][Iter.]: [19.1067s][0.0108s]
    [Ep.][Iter.]: [1][150] | Loss: 0.2540 | Time[Data][Iter.]: [19.3830s][0.0100s]

    => Training
    Epoch: 1 | mAP: 0.0972 | Time(eval): 8.93s
    => Validation
    Epoch: 1 | mAP: 0.1309 | Loss: 0.3584 | Time: 50.94s

    [Ep.][Iter.]: [2][200] | Loss: 0.2420 | Time[Data][Iter.]: [69.6498s][0.8330s]
    [Ep.][Iter.]: [2][250] | Loss: 0.2430 | Time[Data][Iter.]: [19.2637s][0.0091s]
    [Ep.][Iter.]: [2][300] | Loss: 0.2355 | Time[Data][Iter.]: [19.4451s][0.0091s]
    [Ep.][Iter.]: [2][350] | Loss: 0.2363 | Time[Data][Iter.]: [19.4421s][0.0110s]

    => Training
    Epoch: 2 | mAP: 0.1387 | Time(eval): 12.58s
    => Validation
    Epoch: 2 | mAP: 0.1518 | Loss: 0.2554 | Time: 49.55s

    [Ep.][Iter.]: [3][400] | Loss: 0.2318 | Time[Data][Iter.]: [18.9972s][0.8029s]
    [Ep.][Iter.]: [3][450] | Loss: 0.2323 | Time[Data][Iter.]: [19.3416s][0.0094s]
    [Ep.][Iter.]: [3][500] | Loss: 0.2275 | Time[Data][Iter.]: [19.4716s][0.0093s]

    => Training
    Epoch: 3 | mAP: 0.1628 | Time(eval): 12.82s
    => Validation
    Epoch: 3 | mAP: 0.1717 | Loss: 0.2521 | Time: 52.35s

    [Ep.][Iter.]: [4][550] | Loss: 0.2299 | Time[Data][Iter.]: [19.1360s][0.8748s]
    [Ep.][Iter.]: [4][600] | Loss: 0.2243 | Time[Data][Iter.]: [19.4029s][0.0122s]
    [Ep.][Iter.]: [4][650] | Loss: 0.2250 | Time[Data][Iter.]: [19.3620s][0.0105s]
    [Ep.][Iter.]: [4][700] | Loss: 0.2226 | Time[Data][Iter.]: [19.4091s][0.0133s]

    => Training
    Epoch: 4 | mAP: 0.1865 | Time(eval): 8.82s
    => Validation
    Epoch: 4 | mAP: 0.1832 | Loss: 0.2540 | Time: 48.73s

    [Ep.][Iter.]: [5][750] | Loss: 0.2219 | Time[Data][Iter.]: [19.2582s][0.7557s]
    [Ep.][Iter.]: [5][800] | Loss: 0.2180 | Time[Data][Iter.]: [19.3472s][0.0093s]
    [Ep.][Iter.]: [5][850] | Loss: 0.2195 | Time[Data][Iter.]: [19.4909s][0.0093s]

    => Training
    Epoch: 5 | mAP: 0.2040 | Time(eval): 12.79s
    => Validation
    Epoch: 5 | mAP: 0.1996 | Loss: 0.2434 | Time: 49.10s
    """