"""
Test box pair mask construction

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import matplotlib.pyplot as plt

from pocket.ops import BoxPairMultiScaleRoIAlign

def test_2():

    m = BoxPairMultiScaleRoIAlign(
        output_size=4, 
        spatial_scale=[1/8, 1/16, 1/32], 
        sampling_ratio=2
    )

    f = list([
        torch.ones(1, 1, 64, 64),
        torch.ones(1, 1, 32, 32),
        torch.ones(1, 1, 16, 16),
    ])

    boxes_h = torch.tensor([
        [0., 0., 0., 128., 128.],
        [0., 256., 256., 384., 384.],
        [0., 128., 128., 384., 384.]
    ])
    boxes_o = torch.tensor([
        [0., 256., 256., 384., 384.],
        [0., 128., 128., 256., 384.],
        [0., 256., 0., 512., 256.]
    ])

    level = 1
    masks = m.construct_masks_for_box_pairs(f[level], level, boxes_h, boxes_o)

    return masks

def test_1():

    m = BoxPairMultiScaleRoIAlign(7, (1.0,), 2)

    f = torch.rand(2, 1, 12, 12)

    boxes_h = torch.tensor([
        [0.5, 0.5, 5.5, 5.5],
        [1.7, 2.5, 4.9, 4.6]
    ])
    boxes_o = torch.tensor([
        [4.5, 4.5, 9.5, 9.5],
        [4.2, 3.8, 8.7, 7.9]
    ])
    batch_idx = torch.tensor([
        [0.],
        [0.],
        [1.],
        [1.]
    ])

    boxes_h = torch.cat([
        batch_idx,
        torch.cat([boxes_h]*2, 0)
    ], 1)
    boxes_o = torch.cat([
        batch_idx,
        torch.cat([boxes_o]*2, 0)
    ], 1)

    masks = m.construct_masks_for_box_pairs(f, 0, boxes_h, boxes_o)

    return masks

if __name__ == '__main__':
    masks = test_2()
    for mask in masks:
        mask = mask.squeeze()
        plt.imshow(mask)
        plt.colorbar()
        plt.show()