"""
Test bounding box associations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import pocket
import unittest

class TestAssociation(unittest.TestCase):

    def test_duplicates(self):

        association = pocket.utils.BoxAssociation(min_iou=.5)

        boxes_gt = torch.as_tensor([
            [30., 30., 60., 60.],
        ])
        boxes_det = torch.as_tensor([
            [28.8, 31.2, 59.1, 58.4],
            [26.9, 29.2, 63.5, 66.4],
        ])
        scores = torch.as_tensor([
            0.8, 0.9
        ])

        labels_1 = association(boxes_gt, boxes_det, scores)
        labels_2 = association(boxes_gt, boxes_det)

        self.assertEqual(labels_1.tolist(), [0., 1.])
        self.assertEqual(labels_2.tolist(), [1., 0.])

if __name__ == '__main__':
    unittest.main()