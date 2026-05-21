"""Tests para encodings posicionales (sinusoidal + RoPE)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from positional import sinusoidal_encoding, rope_apply


class TestSinusoidalEncoding(unittest.TestCase):
    def test_shape(self):
        enc = sinusoidal_encoding(n_tokens=10, d_model=8)
        self.assertEqual(enc.shape, (10, 8))

    def test_values_bounded(self):
        enc = sinusoidal_encoding(n_tokens=100, d_model=16)
        self.assertTrue(np.all(enc >= -1.0))
        self.assertTrue(np.all(enc <= 1.0))

    def test_position_0_distinct_from_position_1(self):
        enc = sinusoidal_encoding(n_tokens=10, d_model=8)
        self.assertFalse(np.allclose(enc[0], enc[1]))


class TestRoPE(unittest.TestCase):
    def test_shape_preserved(self):
        x = np.random.randn(2, 5, 8)
        out = rope_apply(x)
        self.assertEqual(out.shape, x.shape)

    def test_norm_preserved(self):
        # RoPE es una rotación: debe preservar la norma euclídea de cada token
        np.random.seed(0)
        x = np.random.randn(1, 5, 8)
        out = rope_apply(x)
        norms_in = np.linalg.norm(x, axis=-1)
        norms_out = np.linalg.norm(out, axis=-1)
        np.testing.assert_allclose(norms_in, norms_out, atol=1e-6)

    def test_raises_on_odd_d_model(self):
        x = np.random.randn(1, 5, 7)
        with self.assertRaises(ValueError):
            rope_apply(x)


if __name__ == "__main__":
    unittest.main()
