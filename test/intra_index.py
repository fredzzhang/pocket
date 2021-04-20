"""
Test intra-index computation and 
profile different implementations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Center for Robotic Vision
"""

import torch
import unittest
import torch.autograd.profiler as profiler

from pocket.ops import intra_index

class TestIntraIndex(unittest.TestCase):

    def test_broadcast_correctness(self):
        x = torch.rand(50)
        y = torch.cat([x, torch.rand(50)])
        z = torch.arange(50)
        self.assertTrue(torch.all(z == intra_index(y, x)))

    def test_loop_correctness(self):
        x = torch.rand(50)
        y = torch.cat([x, torch.rand(50)])
        z = torch.arange(50)
        self.assertTrue(torch.all(z == intra_index(y, x, algorithm='loop')))

    def test_broadcast_missing_values(self):
        x = torch.rand(50)
        y = torch.cat([x, torch.rand(50)])
        self.assertRaises(ValueError, intra_index, x, y)

    def test_loop_missing_values(self):
        x = torch.rand(50)
        y = torch.cat([x, torch.rand(50)])
        self.assertRaises(ValueError, intra_index, x, y)

    def test_undefined_algorithm(self):
        x = torch.rand(50)
        y = torch.cat([x, torch.rand(50)])
        self.assertRaises(ValueError, intra_index, y, x, 'boradcast')

    def test_unsupported_shapes(self):
        x = torch.rand(50, 1)
        y = torch.cat([x, torch.rand(50, 1)])
        self.assertRaises(ValueError, intra_index, y, x)

if __name__ == '__main__':

    size = [50, 1000, 20000]
    x_ = [torch.rand(s) for s in size]
    y_ = [torch.cat([x, torch.rand(s)]) for x, s in zip(x_, size)]

    for x, y in zip(x_, y_):
        with profiler.profile(record_shapes=True, profile_memory=True) as prof:
            with profiler.record_function("broadcast"):
                _ = intra_index(y, x)
        print("\n=> Algorithm: broadcast\n")
        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


        with profiler.profile(record_shapes=True, profile_memory=True) as prof:
            with profiler.record_function("loop"):
                _ = intra_index(y, x, algorithm='loop')
        print("\n=> Algorithm: loop\n")
        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


    unittest.main()