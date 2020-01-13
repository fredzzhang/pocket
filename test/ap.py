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
    labels[torch.randperm(100)[:30], 2] = 1     # 30% positives
    labels[torch.randperm(100)[:60], 3] = 1     # 60% positives

    meter = AveragePrecisionMeter(algorithm=alg, output=output, labels=labels)
    # Test on-the-fly result collection
    meter.append(output, labels)

    return meter.eval()

def test_2(alg):
    
    num_detecitons = [100, 150, 200, 250]

    output = [torch.rand(n) for n in num_detecitons]
    labels = [torch.zeros(n) for n in num_detecitons]

    labels[0] += 1
    labels[2][torch.randperm(num_detecitons[2])[:40]] = 1   # 20% positives
    labels[3][torch.randperm(num_detecitons[3])[:50]] = 1   # 20% positives

    meter = DetectionAPMeter(4, algorithm=alg, output=output, labels=labels)
    # Test on-the-fly result collection
    meter.append(
        torch.rand(100), 
        3 * torch.ones(100), 
        torch.randint(-3, 2, (100,)).clamp(0, 1)    # 20% positives on average
    )

    return meter.eval()

def test_3(alg):

    output = torch.rand(200)
    labels = torch.zeros(200)
    labels[torch.randperm(200)[:80]] = 1

    # NOTE: When there is fixed proportion (p) of true positives with a retrieval
    # rate of q, the mathematical expectation of AP is pq
    # In the follwing example, E(AP) = (80 / 200) * (80 * 2 / 200) = 0.32
    meter = DetectionAPMeter(1, num_gt=[200], algorithm=alg)

    meter.append(output, torch.zeros(200), labels)
    meter.append(output, torch.zeros(200), labels)
    # NOTE: Uncommenting the following line should result in error 
    # since the number of true positives exceeds the total number of positives
    # meter.append(output, torch.zeros(200), labels)

    return meter.eval()

if __name__ == '__main__':
    alg = 'AUC'
    niter = 10

    ap = torch.zeros(4)
    for _ in range(niter):
        ap += test_1(alg)
    print(ap / niter)

    ap = torch.zeros(4)
    for _ in range(niter):
        ap += test_2(alg)
    print(ap / niter)

    ap = torch.zeros(1)
    for _ in range(niter):
        ap += test_3(alg)
    print(ap / niter)