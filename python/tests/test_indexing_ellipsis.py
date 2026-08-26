# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestSingleEllipsisIndexing(mlx_tests.MLXTestCase):
    def test_getitem(self):
        a = mx.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(np.array(a[...]), np.array(a[(...,)]))

    def test_setitem_scalar(self):
        a = mx.arange(6).reshape(2, 3)
        a[...] = 42
        np.testing.assert_array_equal(np.array(a), np.full((2, 3), 42))

    def test_setitem_array(self):
        a = mx.zeros((2, 3), dtype=mx.int32)
        a[...] = mx.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(np.array(a), np.arange(6).reshape(2, 3))

    def test_setitem_zero_dimensional(self):
        a = mx.array(1)
        a[...] = 7
        self.assertEqual(a.item(), 7)


if __name__ == "__main__":
    unittest.main()
