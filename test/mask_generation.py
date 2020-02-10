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
            [0., 0., 50., 50.],
            [25., 25., 75., 75.]
        ])
        m = generate_binary_masks(b, 100, 100)
        self.assertEqual(m.sum(), 50 ** 2 * 2)
        self.assertEqual(m[0][:50, :50].sum(), 50 ** 2)
        self.assertEqual(m[1][25:75, 25:75].sum(), 50 ** 2)

    def test_square_masks(self):
        b = torch.tensor([
            [3.2, 8.1, 61.2, 91.5],
            [0., 0., 12.5, 100.3],
            [45.1, 2.8, 120., 120.],
            [31.6, 71.2, 52.9, 101.4]
        ])
        m = generate_binary_masks(b, 120, 120)
        area = (b[:, 3] - b[:, 1]) * (b[:, 2] - b[:, 0])
        self.assertEqual(m.sum(), area.sum())

    def test_rectangular_masks(self):
        b = torch.tensor([
            [31.52, 192.37, 8.15, 100.2],
            [0.6, 0.2, 319.8, 158.9],
            [39.19, 81.25, 145.9, 141.85]
        ])
        m = generate_binary_masks(b, 160, 310)
        area = (b[:, 3] - b[:, 1]) * (b[:, 2] - b[:, 0])
        self.assertEqual(m.sum(), area.sum())

if __name__ == '__main__':
    unittest.main()