"""Tests para multi-head attention."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from multi_head import multi_head_attention


class TestMultiHeadAttention(unittest.TestCase):
    def test_output_shape(self):
        np.random.seed(42)
        batch, n, d_model, n_heads = 1, 6, 16, 4
        X = np.random.randn(batch, n, d_model)
        Wq = np.random.randn(d_model, d_model)
        Wk = np.random.randn(d_model, d_model)
        Wv = np.random.randn(d_model, d_model)
        Wo = np.random.randn(d_model, d_model)
        out = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=n_heads)
        self.assertEqual(out.shape, (batch, n, d_model))

    def test_raises_when_d_model_not_divisible_by_n_heads(self):
        X = np.random.randn(1, 6, 17)
        W = np.random.randn(17, 17)
        with self.assertRaises(ValueError):
            multi_head_attention(X, W, W, W, W, n_heads=4)

    def test_single_head_equivalent_to_scaled_dot_product(self):
        # Con n_heads=1 y proyecciones identidad, debe equivaler a SDPA simple
        from attention import scaled_dot_product_attention
        np.random.seed(0)
        d_model = 8
        n = 5
        X = np.random.randn(1, n, d_model)
        I = np.eye(d_model)
        out_mha = multi_head_attention(X, I, I, I, I, n_heads=1)
        out_sdpa, _ = scaled_dot_product_attention(X, X, X)
        np.testing.assert_allclose(out_mha, out_sdpa, atol=1e-6)

    def test_multiple_heads_produce_different_output_than_single(self):
        np.random.seed(0)
        batch, n, d_model = 1, 6, 16
        X = np.random.randn(batch, n, d_model)
        W = np.random.randn(d_model, d_model)
        out_1 = multi_head_attention(X, W, W, W, W, n_heads=1)
        out_4 = multi_head_attention(X, W, W, W, W, n_heads=4)
        # No deben ser idénticos (las proyecciones se reparten entre cabezales)
        self.assertFalse(np.allclose(out_1, out_4))


if __name__ == "__main__":
    unittest.main()
