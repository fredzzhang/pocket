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
    => Validation (+9.74s)
    Epoch: 0 | Acc.: 0.1008[1008/10000] | Loss: 2.3036 | Time: 1.83s

    [Ep.][Iter.]: [1][100] | Loss: 2.2971 | Time[Data/Iter.]: [2.8266s/3.1294s]
    [Ep.][Iter.]: [1][200] | Loss: 2.2773 | Time[Data/Iter.]: [2.6130s/2.9324s]
    [Ep.][Iter.]: [1][300] | Loss: 2.2289 | Time[Data/Iter.]: [2.3087s/2.6148s]
    [Ep.][Iter.]: [1][400] | Loss: 2.0142 | Time[Data/Iter.]: [2.1537s/2.4501s]

    => Training (+22.64s)
    Epoch: 1 | Acc.: 0.3182[19091/60000]
    => Validation (+24.36s)
    Epoch: 1 | Acc.: 0.7949[7949/10000] | Loss: 0.7700 | Time: 1.72s
    """
