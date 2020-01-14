"""
Test masked box pair pooling

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import time
import torch
from torchvision.ops import roi_align
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pocket.ops import MaskedBoxPairPool

def test_1():
    """Visualise the pooled box pair feautures"""

    image = torch.ones(512, 512, 3).numpy()

    f = list([
        torch.ones(1, 3, 64, 64),
        torch.ones(1, 3, 32, 32),
        torch.ones(1, 3, 16, 16),
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

    m = MaskedBoxPairPool(
        output_size=5, 
        spatial_scale=[1/8, 1/16, 1/32], 
        sampling_ratio=2
    )

    t = time.time()
    out = m(f, [boxes_h], [boxes_o]).permute([0, 2, 3, 1])
    print(time.time() - t)

    num_boxes = len(boxes_h)
    for idx in range(num_boxes):
        ax = plt.subplot(num_boxes, 2, idx * 2 + 1)
        ax.imshow(image)
        box_h = boxes_h[idx].numpy()
        box_h[2:] -= box_h[:2]
        rect = Rectangle(box_h[:2], box_h[2], box_h[3], fill=False, edgecolor='k')
        ax.add_patch(rect)
        box_o = boxes_o[idx].numpy()
        box_o[2:] -= box_o[:2]
        rect = Rectangle(box_o[:2], box_o[2], box_o[3], fill=False, edgecolor='k')
        ax.add_patch(rect)

        plt.subplot(num_boxes, 2, idx * 2 + 2)
        plt.imshow(out[idx].squeeze().numpy())
    plt.show()

def test_2():
    """Authenticate the pooled box pair features """
    f = torch.rand(1, 3, 512, 512)

    boxes_h = torch.rand(256, 4) * 256; boxes_h[:, 2:] += boxes_h[:, :2]
    boxes_h = torch.cat([torch.zeros(256, 1), boxes_h], 1)
    boxes_o = torch.rand(256, 4) * 256; boxes_o[:, 2:] += boxes_o[:, :2]
    boxes_o = torch.cat([torch.zeros(256, 1), boxes_o], 1)

    boxes_union = torch.zeros_like(boxes_h)
    boxes_union[:, 1] = torch.min(boxes_h[:, 1], boxes_o[:, 1])
    boxes_union[:, 2] = torch.min(boxes_h[:, 2], boxes_o[:, 2])
    boxes_union[:, 3] = torch.max(boxes_h[:, 3], boxes_o[:, 3])
    boxes_union[:, 4] = torch.max(boxes_h[:, 4], boxes_o[:, 4])

    m = MaskedBoxPairPool(
        output_size=7,
        spatial_scale=[1.0],
        sampling_ratio=4
    )
    # Compute pooled box pair features
    out1 = m([f], [boxes_h[:, 1:]], [boxes_o[:, 1:]])

    masks = m.construct_masks_for_box_pairs(f, 0, boxes_h, boxes_o)
    # Apply masks on feature maps
    f_stacked = f[boxes_union[:, 0].long()] * masks
    boxes_union[:, 0] = torch.arange(256)
    # Compute pooled box union features
    out2 = roi_align(f_stacked, boxes_union, 
        output_size=(7,7), spatial_scale=1.0, sampling_ratio=4)

    # Compare the pooled features
    # The two feature maps should be exactly the same
    assert out1.shape == out2.shape, \
        "Inconsistent feature map size"
    print("Feature maps are {}% matched.".format(
        100 * torch.eq(out1, out2).sum() / torch.as_tensor(out1.shape).prod()))

if __name__ == '__main__':
    test_1()
    # test_2()
