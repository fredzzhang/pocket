"""
Test box pair pooling

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import time
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pocket.ops import BoxPairMultiScaleRoIAlign

def test():

    image = torch.ones(512, 512).numpy()

    f = list([
        torch.ones(1, 1, 64, 64),
        torch.ones(1, 1, 32, 32),
        torch.ones(1, 1, 16, 16),
    ])

    boxes_h = torch.tensor([
        [0., 0., 128., 128.],
        [256., 256., 384., 384.],
        [128., 128., 384., 384.]
    ])
    boxes_o = torch.tensor([
        [256., 256., 384., 384.],
        [128., 128., 256., 384.],
        [256., 0., 512., 256.]
    ])

    m = BoxPairMultiScaleRoIAlign(output_size=4, spatial_scale=[1/8, 1/16, 1/32], sampling_ratio=2)

    t = time.time()
    out = m(f, [boxes_h], [boxes_o])
    print(time.time() - t)

    num_boxes = len(boxes_h)
    for idx in range(num_boxes):
        ax = plt.subplot(num_boxes, 2, idx * 2 + 1)
        ax.imshow(image)
        box_h = boxes_h[idx].numpy()
        box_h[2:] -= box_h[:2]
        rect = Rectangle(box_h[:2], box_h[2], box_h[3], fill=False, edgecolor='w')
        ax.add_patch(rect)
        box_o = boxes_o[idx].numpy()
        box_o[2:] -= box_o[:2]
        rect = Rectangle(box_o[:2], box_o[2], box_o[3], fill=False, edgecolor='w')
        ax.add_patch(rect)

        plt.subplot(num_boxes, 2, idx * 2 + 2)
        plt.imshow(out[idx].squeeze().numpy())
    plt.show()

if __name__ == '__main__':
    test()