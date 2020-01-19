"""
Test Sinkhorn-Knopp algorithm

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Center for Robotic Vision
"""

import torch
import unittest
import warnings

from pocket.ops import SinkhornKnoppNorm

class TestSinkhornKnopp(unittest.TestCase):

    def test_basic(self, m=None):
        if m is None:
            self.skipTest("Module to be tested is left as None")

        self.assertIsNotNone(m)
        self.assertEqual(m.niter, None)
        self.assertEqual(m.max_iter, 1000)
        self.assertEqual(m.tolerance, 0.001)

        self.assertRaises(AssertionError, SinkhornKnoppNorm, max_iter='10')
        self.assertRaises(AssertionError, SinkhornKnoppNorm, max_iter=-1)
        self.assertRaises(AssertionError, SinkhornKnoppNorm, tolerance='0.5')
        self.assertRaises(AssertionError, SinkhornKnoppNorm, tolerance=0)
        with self.assertRaises(AttributeError):
            m.tolerance = 0.01
        with self.assertRaises(AttributeError):
            m.niter = 0

        m.max_iter = 200
        self.assertEqual(m.max_iter, 200)

    def test_init(self):
        m = SinkhornKnoppNorm()
        n = eval(repr(m))
        self.test_basic(m)
        self.test_basic(n)

    def test_random_logits(self):
        m = SinkhornKnoppNorm()
        x = m(torch.rand(6, 6))
        self.assertTrue(torch.all((x.sum(0) - 1).abs() < m.tolerance))
        self.assertTrue(torch.all((x.sum(1) - 1).abs() < m.tolerance))

        x = m(torch.rand(3000, 3000))
        self.assertTrue(torch.all((x.sum(0) - 1).abs() < m.tolerance))
        self.assertTrue(torch.all((x.sum(1) - 1).abs() < m.tolerance))

    def test_input_format(self):
        m = SinkhornKnoppNorm()
        x = torch.tensor([[1, 2], [-1, 3]])
        self.assertRaises(AssertionError, m, x)
        x = torch.rand(3,4,5)
        self.assertRaises(AssertionError, m, x)

        x = torch.randint(0, 10, (30, 30))
        x1 = m(x.numpy())
        self.assertTrue
        self.assertTrue(torch.all((x1.sum(0) - 1).abs() < m.tolerance))
        self.assertTrue(torch.all((x1.sum(1) - 1).abs() < m.tolerance))
        x2 = m(x.tolist())
        self.assertTrue(torch.all((x2.sum(0) - 1).abs() < m.tolerance))
        self.assertTrue(torch.all((x2.sum(1) - 1).abs() < m.tolerance))

    def test_warnings(self):
        m = SinkhornKnoppNorm()
        with warnings.catch_warnings(record=True) as w:
            x = m(torch.cat([
                torch.rand(5, 5),
                torch.zeros(5, 1)
            ], 1))
            for item in w:
                self.assertEqual(item.category, UserWarning)

        self.assertTrue(torch.all((x.sum(0)[:-1] - 1).abs() < m.tolerance))
        self.assertTrue(torch.all((x.sum(1) - 1).abs() < m.tolerance))

if __name__ == '__main__':
    unittest.main()