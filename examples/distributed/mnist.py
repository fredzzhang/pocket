"""
An example on MNIST handwritten digits recognition
This script uses DistributedDataParallel

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms

from pocket.models import LeNet
from pocket.core import DistributedLearningEngine

def main(rank, world_size):
    # Initialisation
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    # Fix random seed
    torch.manual_seed(0)
    # Initialize network
    net = LeNet()
    # Initialize loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Prepare dataset
    trainset = datasets.MNIST('../data', train=True, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    )
    # Prepare sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=rank
    )
    # Prepare dataloader
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=False,
        num_workers=2, pin_memory=True, sampler=train_sampler)
    # Intialize learning engine and start training
    engine = DistributedLearningEngine(
        net, criterion, train_loader,
    )
    # Train the network for one epoch with default optimizer option
    # Checkpoints will be saved under ./checkpoints by default, containing 
    # saved model parameters, optimizer statistics and progress
    engine(5)

    # Clean up
    dist.destroy_process_group()

if __name__ == '__main__':

    # Number of GPUs to run the experiment with
    WORLD_SIZE = 2

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    mp.spawn(main, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))

    # Sample output
    """
    Epoch [1/5], Iter. [100/235], Loss: 2.2968, Time[Data/Iter.]: [3.31s/6.57s]
    Epoch [1/5], Iter. [200/235], Loss: 2.2767, Time[Data/Iter.]: [2.30s/5.07s]
    Epoch [2/5], Iter. [065/235], Loss: 2.2289, Time[Data/Iter.]: [3.13s/5.50s]
    Epoch [2/5], Iter. [165/235], Loss: 2.0091, Time[Data/Iter.]: [2.11s/4.99s]
    Epoch [3/5], Iter. [030/235], Loss: 1.0353, Time[Data/Iter.]: [3.21s/5.81s]
    Epoch [3/5], Iter. [130/235], Loss: 0.5111, Time[Data/Iter.]: [2.59s/5.80s]
    Epoch [3/5], Iter. [230/235], Loss: 0.4194, Time[Data/Iter.]: [2.32s/5.14s]
    Epoch [4/5], Iter. [095/235], Loss: 0.3574, Time[Data/Iter.]: [3.01s/5.64s]
    Epoch [4/5], Iter. [195/235], Loss: 0.3105, Time[Data/Iter.]: [2.39s/4.99s]
    Epoch [5/5], Iter. [060/235], Loss: 0.2800, Time[Data/Iter.]: [3.23s/6.19s]
    Epoch [5/5], Iter. [160/235], Loss: 0.2575, Time[Data/Iter.]: [2.44s/4.67s]
    """
