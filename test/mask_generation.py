"""
Test binary masks generation

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import unittest

from pocket.ops import generate_masks

class TestMaskGeneration(unittest.TestCase):

    def test_integer_coords(self):
        b = torch.tensor([
            [0., 0., 50., 50.],
            [25., 25., 75., 75.]
        ])
        m = generate_masks(b, 100, 100)
        self.assertEqual(m.sum(), 50 ** 2 * 2)
        self.assertEqual(m[0][:50, :50].sum(), 50 ** 2)
        self.assertEqual(m[1][25:75, 25:75].sum(), 50 ** 2)

        # Float type has a precision of 7 decimal places
        eps = 1e-7
        b = torch.tensor([
            [21.5, 9.3, 72.6, 51.3]
        ])
        m = generate_masks(b, 60, 80)
        self.assertEqual(m.sum(), (b[0, 3] - b[0, 1]) * (b[0, 2] - b[0, 0]))
        self.assertTrue(m[0, 9, 21].item() - 0.35 < eps)
        self.assertTrue(m[0, 9, 72].item() - 0.42 < eps)
        self.assertTrue(m[0, 51, 21].item() - 0.15 < eps)
        self.assertTrue(m[0, 51, 72].item() - 0.18 < eps)

    def test_square_masks(self):
        size = (120, 120)
        b = torch.tensor([
            [3.2, 8.1, 61.2, 91.5],
            [0., 0., 12.5, 100.3],
            [45.1, 18.8, 120., 120.],
            [33.6, 41.2, 52.9, 101.4],
        ])
        m = generate_masks(b, *size)
        area = (b[:, 3] - b[:, 1]) * (b[:, 2] - b[:, 0])
        self.assertEqual(m.sum(), area.sum())

    def test_rectangular_masks(self):
        size = (160, 310)
        b = torch.tensor([
            [31.57, 92.37, 182.18, 100.72],
            [1.37, 9.42, 309.38, 158.29],
            [39.29, 81.25, 145.92, 141.81],
        ])
        m = generate_masks(b, *size)
        area = (b[:, 3] - b[:, 1]) * (b[:, 2] - b[:, 0])
        self.assertEqual(m.sum(), area.sum())

    def test_small_boxes(self):
        size = (5, 5)
        b = torch.tensor([
            [0.12, 0.23, 0.58, 0.89],
            [4.1, 3.9, 5.0, 4.8],
            [2.3, 3.4, 2.91, 3.82]
        ])
        m = generate_masks(b, *size)
        area = (b[:, 3] - b[:, 1]) * (b[:, 2] - b[:, 0])
        self.assertEqual(m.sum(), area.sum())

    def test_random_coords(self):
        for _ in range(100):
            h, w = torch.randint(50, 650, (2,))
            x1, y1, w_, h_ = torch.rand(1, 4).unbind(1)
            x1 *= w; y1 *= h
            w_ *= (w-x1); h_ *= (h-y1)
            x2 = x1 + w_; y2 = y1 + h_
            b = torch.stack([x1, y1, x2, y2], 1)
            m = generate_masks(b, h.item(), w.item())
            area = h_ * w_
            # Sum over masks should equal bounding box area up to some
            # numerical error depending on the size of the box
            self.assertTrue(m.sum() - area < 1e-2)

def profiler(num=10000, size=256):
    boxes = torch.rand(num, 4) * size
    boxes = boxes.sort(1)[0]

    with torch.autograd.profiler.profile() as prof:
        masks = generate_masks(boxes, size, size)
    print(prof.key_averages().table(sort_by='cpu_time_total'))

if __name__ == '__main__':
    unittest.main()
    profiler()