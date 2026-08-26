import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


@unittest.skipUnless(mx.metal.is_available(), "Metal is required")
class TestComplexExtremeUnary(mlx_tests.MLXTestCase):
    def test_complex64_abs_extreme_range(self):
        values = np.array(
            [
                1.0e20 + 1.0e20j,
                2.5e-20 - 3.0e-20j,
                3.0e38 + 1.0e38j,
            ],
            dtype=np.complex64,
        )
        expected = np.abs(values)
        actual = np.array(mx.abs(mx.array(values), stream=mx.gpu))
        self.assertTrue(np.allclose(actual, expected, rtol=2e-6, atol=0.0))

    def test_complex64_sqrt_extreme_range(self):
        values = np.array(
            [
                1.0e20 + 1.0e20j,
                2.5e-20 - 3.0e-20j,
                3.0e38 + 3.0e38j,
                -3.0e38 + 3.0e38j,
            ],
            dtype=np.complex64,
        )
        expected = np.sqrt(values)
        actual = np.array(mx.sqrt(mx.array(values), stream=mx.gpu))
        self.assertTrue(np.allclose(actual, expected, rtol=3e-6, atol=0.0))

    def test_complex64_log_extreme_range(self):
        values = np.array(
            [
                1.0e20 + 1.0e20j,
                2.5e-20 - 3.0e-20j,
                3.0e38 + 3.0e38j,
                -3.0e38 + 3.0e38j,
            ],
            dtype=np.complex64,
        )
        expected = np.log(values)
        actual = np.array(mx.log(mx.array(values), stream=mx.gpu))
        self.assertTrue(np.allclose(actual, expected, rtol=2e-6, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
