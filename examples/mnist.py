"""
An example on MNIST handwritten digits recognition

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
from torchvision import datasets, transforms
from pocket.models import LeNet
from pocket.utils import MultiClassClassificationEngine

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
        datasets.MNIST('./data', train=False, download=True,
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
    => Validation
    Epoch: 0 | Acc.: 0.1008[1008/10000] | Loss: 2.3036 | Time: 5.24s

    [Ep.][Iter.]: [1][100] | Loss: 1.1665 | Time[Data][Iter.]: [466.5847s][466.5847s]
    [Ep.][Iter.]: [1][200] | Loss: 2.2773 | Time[Data][Iter.]: [0.2861s][2.6689s]
    [Ep.][Iter.]: [1][300] | Loss: 2.2289 | Time[Data][Iter.]: [0.2581s][2.5305s]
    [Ep.][Iter.]: [1][400] | Loss: 2.0143 | Time[Data][Iter.]: [0.2986s][2.6412s]

    => Training
    Epoch: 1 | Acc.: 0.3181[19087/60000]
    => Validation
    Epoch: 1 | Acc.: 0.7950[7950/10000] | Loss: 1.0872 | Time: 2.18s
    """