"""
Test binary masks generation

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import unittest

from pocket.ops import generate_binary_masks

class TestMaskGeneration(unittest.TestCase):

    def test_integer_coords(self):
        b = torch.tensor([
            [0, 0, 50, 50],
            [25, 25, 75, 75]
        ])
        m = generate_binary_masks(b, 100, 100)
        self.assertEqual(m.sum(), 50 ** 2 * 2)
        self.assertEqual(m[0][:50, :50].sum(), 50 ** 2)
        self.assertEqual(m[1][25:75, 25:75].sum(), 50 ** 2)

    def test_random_squares(self):
        for _ in range(20):
            b = torch.rand(50, 4) * 120; b[:, 2:] += b[:, :2]
            m = generate_binary_masks(b, 240, 240)
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            self.assertEqual(m.sum(), area.sum())

    def test_random_rectangles(self):
        for _ in range(20):
            h, w = torch.randint(50, 150, (2,))
            x1, y1, w_, h_ = torch.rand(100, 4).unbind(1)
            x1 *= w; y1 *= h
            w_ *= (w-x1); h_ *= (h-y1)
            x2 = x1 + w_; y2 = y1 + h_
            b = torch.stack([x1, y1, x2, y2], 1)

            m = generate_binary_masks(b, h, w)
            area = h_ * w_
            self.assertEqual(m.sum(), area.sum())

if __name__ == '__main__':
    unittest.main()