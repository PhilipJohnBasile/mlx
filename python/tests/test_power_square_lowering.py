import numpy as np

import mlx.core as mx


def test_scalar_two_power_lowers_to_square():
    rng = np.random.default_rng(42)
    for dtype, np_dtype in ((mx.float32, np.float32), (mx.float64, np.float64)):
        values = rng.uniform(-3.0, 3.0, 4096).astype(np_dtype)
        x = mx.array(values, dtype=dtype)
        expected = np.asarray(mx.square(x))
        assert np.array_equal(np.asarray(x ** 2), expected)
        assert np.array_equal(np.asarray(mx.power(x, 2)), expected)


def test_nonliteral_power_keeps_general_path_semantics():
    x = mx.array([1.25, 2.0, 3.5], dtype=mx.float32)
    exponent = mx.add(mx.array(1.0), mx.array(1.0))
    np.testing.assert_allclose(
        np.asarray(mx.power(x, exponent)),
        np.asarray(mx.power(x, mx.array(2.0))),
    )
