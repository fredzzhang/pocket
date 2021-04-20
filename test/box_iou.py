"""
Test box IoU computation

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Center for Robotic Vision
"""

import torch
import unittest

from pocket.ops import box_iou

class TestBoxIoU(unittest.TestCase):

    def test_compatibility(self):
        boxes_1 = torch.rand(100, 4) * 256
        boxes_1[:, 2:] += boxes_1[:, :2]
        boxes_2 = torch.rand(100, 4) * 256
        boxes_2[:, 2:] += boxes_2[:, :2]
        iou_1 = box_iou(boxes_1, boxes_2, encoding='coord')

        boxes_1[:, 2:] -= 1; boxes_2[:, 2:] -= 1
        iou_2 = box_iou(boxes_1, boxes_2, encoding='pixel')

        self.assertTrue(torch.all(iou_1 == iou_2))

if __name__ == '__main__':
    unittest.main()