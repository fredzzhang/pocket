"""
An example on MNIST handwritten digits recognition

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
from torchvision import datasets, transforms
from pocket.models import LeNet
from pocket.core import MultiClassClassificationEngine

def main():
    # Fix random seed
    torch.manual_seed(0)
    # Initialize network
    net = LeNet()
    # Initialize loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Prepare dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            ),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            ),
        batch_size=100, shuffle=False)
    # Intialize learning engine and start training
    engine = MultiClassClassificationEngine(net, criterion, train_loader,
        val_loader=test_loader)
    # Train the network for one epoch with default optimizer option
    # Checkpoints will be saved under ./checkpoints by default, containing 
    # saved model parameters, optimizer statistics and progress
    engine(1)

if __name__ == '__main__':
    main()

    # Sample output
    """
    => Validation (+3.67s)
    Epoch: 0 | Acc.: 0.1008[1008/10000] | Loss: 2.3036 | Time: 1.99s

    Epoch [1/1], Iter. [100/469], Loss: 2.2971, Time[Data/Iter.]: [1.73s/1.98s]
    Epoch [1/1], Iter. [200/469], Loss: 2.2773, Time[Data/Iter.]: [1.70s/1.96s]
    Epoch [1/1], Iter. [300/469], Loss: 2.2289, Time[Data/Iter.]: [1.68s/1.98s]
    Epoch [1/1], Iter. [400/469], Loss: 2.0143, Time[Data/Iter.]: [1.69s/1.99s]

    => Training (+12.96s)
    Epoch: 1 | Acc.: 0.3182[19090/60000]
    => Validation (+14.65s)
    Epoch: 1 | Acc.: 0.7947[7947/10000] | Loss: 0.7701 | Time: 1.69s
    """
