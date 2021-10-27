"""
Test NMS on bounding boxes and box pairs

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Center for Robotic Vision
"""

import torch
import torchvision

import pocket
import unittest

class TestNMS(unittest.TestCase):

    def test_nms_validity_random(self):
        n = 100
        boxes = torch.rand(n, 4) * 100
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.rand(n)

        k1 = torchvision.ops.boxes.nms(boxes, scores, 0.5)
        k2 = pocket.ops.nms(boxes, scores, 0.5)
        self.assertTrue(torch.all(k1.eq(k2)))

    def test_nms_validity_duplicates(self):
        boxes = torch.rand(1, 4) * 100
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.rand(1)

        boxes = boxes.repeat(100, 1)
        scores = scores.repeat(100)

        k1 = torchvision.ops.boxes.nms(boxes, scores, 0.5)
        k2 = pocket.ops.nms(boxes, scores, 0.5)
        self.assertTrue(torch.all(k1.eq(k2)))

if __name__ == '__main__':
    unittest.main()