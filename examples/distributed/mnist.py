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
        rank, rank, world_size,
        net, criterion, train_loader,
    )
    # Train the network for one epoch with default optimizer option
    # Checkpoints will be saved under ./checkpoints by default, containing 
    # saved model parameters, optimizer statistics and progress
    engine(5)

if __name__ == '__main__':

    # Number of GPUs to run the experiment with
    WORLD_SIZE = 2

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    mp.spawn(main, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))
