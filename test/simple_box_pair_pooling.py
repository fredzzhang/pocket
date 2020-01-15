"""
Test box pair pooling

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import time
import torch
from collections import OrderedDict
from torchvision.ops import MultiScaleRoIAlign

from pocket.ops import SimpleBoxPairPool

def test():
    
    # Image size
    im_s = 1024

    f = list([
        torch.rand(1, 3, 128, 128),
        torch.rand(1, 3, 64, 64),
        torch.rand(1, 3, 32, 32),
    ])

    pts = torch.rand(256, 4) * im_s

    boxes_h = torch.zeros(128, 4)
    boxes_h[:, 0] = torch.min(pts[:128, 0], pts[:128, 2])
    boxes_h[:, 1] = torch.min(pts[:128, 1], pts[:128, 3])
    boxes_h[:, 2] = torch.max(pts[:128, 0], pts[:128, 2])
    boxes_h[:, 3] = torch.max(pts[:128, 1], pts[:128, 3])

    boxes_o = torch.zeros(128, 4)
    boxes_o[:, 0] = torch.min(pts[128:, 0], pts[128:, 2])
    boxes_o[:, 1] = torch.min(pts[128:, 1], pts[128:, 3])
    boxes_o[:, 2] = torch.max(pts[128:, 0], pts[128:, 2])
    boxes_o[:, 3] = torch.max(pts[128:, 1], pts[128:, 3])

    m1 = SimpleBoxPairPool(
        output_size=7,
        spatial_scale=[1/8, 1/16, 1/32],
        sampling_ratio=2
    )
    # Compute pooled box pair features
    out1 = m1(f, [boxes_h], [boxes_o])

    boxes_union = boxes_h.clone()
    boxes_union[:, 0] = torch.min(boxes_h[:, 0], boxes_o[:, 0])
    boxes_union[:, 1] = torch.min(boxes_h[:, 1], boxes_o[:, 1])
    boxes_union[:, 2] = torch.max(boxes_h[:, 2], boxes_o[:, 2])
    boxes_union[:, 3] = torch.max(boxes_h[:, 3], boxes_o[:, 3])

    f = OrderedDict([
        (0, f[0]),
        (1, f[1]),
        (2, f[2])
    ])
    m2 = MultiScaleRoIAlign(
        [0, 1, 2], 
        output_size=7, 
        sampling_ratio=2
    )
    # Compute pooled box union features
    out2 = m2(f, [boxes_union], [(im_s, im_s)])

    # Compare the pooled features
    # The two feature maps will be exactly the same when rois are mapped
    # to the same level
    # To do this, change line170 in pocket.ops.SimpleBoxPairPool
    # - levels = self.map_levels(boxes_1, boxes_2)
    # + levels = self.map_levels(box_pair_union)
    assert out1.shape == out2.shape, \
        "Inconsistent feature map size"
    print("Pixels matched: {}/{}.".format(
        torch.eq(out1, out2).sum(), torch.as_tensor(out1.shape).prod()))

if __name__ == '__main__':
    test()
