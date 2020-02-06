"""
Test Sinkhorn-Knopp algorithm

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Center for Robotic Vision
"""

import torch
import unittest

from pocket.ops import SinkhornKnoppNorm2d

class TestSinkhornKnopp(unittest.TestCase):

    def test_basic(self, m=None):
        if m is None:
            self.skipTest("Module to be tested is left as None")

        self.assertIsNotNone(m)
        self.assertEqual(m.niter, None)
        self.assertEqual(m.max_iter, 1000)
        self.assertEqual(m.tolerance, 0.001)

        self.assertRaises(AssertionError, SinkhornKnoppNorm2d, max_iter='10')
        self.assertRaises(AssertionError, SinkhornKnoppNorm2d, max_iter=-1)
        self.assertRaises(AssertionError, SinkhornKnoppNorm2d, tolerance='0.5')
        self.assertRaises(AssertionError, SinkhornKnoppNorm2d, tolerance=0)
        with self.assertRaises(AttributeError):
            m.tolerance = 0.01
        with self.assertRaises(AttributeError):
            m.niter = 0

        m.max_iter = 200
        self.assertEqual(m.max_iter, 200)

    def test_init(self):
        m = SinkhornKnoppNorm2d()
        n = eval(repr(m))
        self.test_basic(m)
        self.test_basic(n)

    def test_random_logits(self):
        m = SinkhornKnoppNorm2d()
        x = m(torch.rand(6, 6))
        self.assertTrue(torch.all((x.sum(0) - 1).abs() < m.tolerance))
        self.assertTrue(torch.all((x.sum(1) - 1).abs() < m.tolerance))

        x = m(torch.rand(3000, 3000))
        self.assertTrue(torch.all((x.sum(0) - 1).abs() < m.tolerance))
        self.assertTrue(torch.all((x.sum(1) - 1).abs() < m.tolerance))

    def test_input_format(self):
        m = SinkhornKnoppNorm2d()
        x = torch.tensor([[1, 2], [-1, 3]])
        self.assertRaises(AssertionError, m, x)
        x = torch.rand(3,4,5)
        self.assertRaises(AssertionError, m, x)
        x = torch.rand(100)
        self.assertRaises(AssertionError, m, x)

        x = torch.randint(0, 10, (30, 30))
        x1 = m(x.numpy())
        self.assertTrue(torch.all((x1.sum(0) - 1).abs() < m.tolerance))
        self.assertTrue(torch.all((x1.sum(1) - 1).abs() < m.tolerance))
        x2 = m(x.tolist())
        self.assertTrue(torch.all((x2.sum(0) - 1).abs() < m.tolerance))
        self.assertTrue(torch.all((x2.sum(1) - 1).abs() < m.tolerance))

    def test_nonsquare_matrices(self):
        m = SinkhornKnoppNorm2d()
        x = m(torch.rand(100, 1000) * 100)
        self.assertTrue(torch.all((x.sum(0) - 0.1).abs() < m.tolerance))
        self.assertTrue(torch.all((x.sum(1) - 1).abs() < m.tolerance))

        x = m(torch.rand(3000, 150) * 50)
        self.assertTrue(torch.all((x.sum(0) - 1).abs() < m.tolerance))
        self.assertTrue(torch.all((x.sum(1) - 0.05).abs() < m.tolerance))

    def test_zero_rows_and_cols(self):
        m = SinkhornKnoppNorm2d()
        x = torch.rand(30, 30)
        r_idx = [0, 15, 16]; c_idx = [2, 9, 28]
        x[r_idx, :] = 0; x[:, c_idx] = 0
        x = m(x)
        self.assertTrue(x.sum() / 30 - 1 < m.tolerance)
        all_idx = list(range(30))
        r_keep = [i for i in all_idx if i not in r_idx]
        c_keep = [i for i in all_idx if i not in c_idx]
        self.assertTrue(torch.all(
            (x.sum(0)[c_keep] - 1).abs() < m.tolerance))
        self.assertTrue(torch.all(
            (x.sum(1)[r_keep] - 1).abs() < m.tolerance))

        x = torch.rand(8, 13)
        # The matrix is essentially 6x11
        x[:2, :] = 0; x[:, :2] = 0
        x = m(x)
        self.assertTrue(x.sum() / 6 - 1 < m.tolerance)
        self.assertTrue(torch.all((x.sum(0)[2:] - 6/11).abs() < m.tolerance))
        self.assertTrue(torch.all((x.sum(1)[2:] - 1).abs() < m.tolerance))

    def test_unusual_shapes(self):
        m = SinkhornKnoppNorm2d()
        x = torch.empty(0, 10)
        self.assertTrue(torch.all(torch.eq(x, m(x))))

        x = torch.zeros(3,5)
        self.assertTrue(torch.all(torch.eq(x, m(x))))

        x = m(torch.rand(1,5))
        self.assertTrue((torch.sum(x) - 1).abs() < m.tolerance)

        x = m(torch.rand(20, 1))
        self.assertTrue((torch.sum(x) - 1).abs() < m.tolerance)

        x = m(torch.rand(1, 1))
        self.assertTrue((x - 1).abs() < m.tolerance)


if __name__ == '__main__':
    unittest.main()
