"""
Test average preicision evaluation

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
from pocket.utils import AveragePrecisionMeter, DetectionAPMeter

def test_1(alg):

    output = torch.rand(100, 4)
    labels = torch.zeros_like(output)

    labels[:, 0] = 1
    # NOTE: The AP of a class with a fixed proportion (p) of true positives randomly
    # selected from the samples has a mathematical expecation of p when computing 
    # strict AUC. For algorithms with interpolation, the expectation is slightly higher
    labels[torch.randperm(100)[:30], 2] = 1
    labels[torch.randperm(100)[:60], 3] = 1

    meter = AveragePrecisionMeter(algorithm=alg, output=output, labels=labels)
    # Test on-the-fly result collection
    meter.append(output, labels)

    return meter.eval()

def test_2(alg):
    
    output = [
        torch.rand(100),
        torch.rand(200),
        torch.rand(150),
        torch.rand(250),
    ]
    labels = [
        torch.zeros(100),
        torch.ones(200),
        torch.randint(0, 3, (150,)).clamp(0, 1),    # 66.7% positives
        torch.randint(0, 4, (250,)).clamp(0, 1),    # 75% positives
    ]

    meter = DetectionAPMeter(4, algorithm=alg, output=output, labels=labels)
    # Test on-the-fly result collection
    meter.append(
        torch.rand(100), 
        3 * torch.ones(100), 
        torch.randint(0, 4, (100,)).clamp(0, 1)
    )

    return meter.eval()

if __name__ == '__main__':
    alg = 'AUC'
    niter = 100

    ap = torch.zeros(4)
    for _ in range(niter):
        ap += test_1(alg)
    print(ap / niter)

    ap = torch.zeros(4)
    for _ in range(niter):
        ap += test_2(alg)
    print(ap / niter)