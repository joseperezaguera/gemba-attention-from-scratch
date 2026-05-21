"""Tests para atención escalar producto."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from attention import softmax, scaled_dot_product_attention


class TestSoftmax(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        out = softmax(x)
        self.assertAlmostEqual(out.sum(), 1.0, places=6)

    def test_softmax_is_numerically_stable_for_large_inputs(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        out = softmax(x)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(out.sum(), 1.0, places=6)

    def test_softmax_along_axis(self):
        x = np.random.randn(3, 4)
        out = softmax(x, axis=-1)
        self.assertTrue(np.allclose(out.sum(axis=-1), 1.0))


class TestScaledDotProductAttention(unittest.TestCase):
    def test_output_shapes(self):
        np.random.seed(42)
        batch, n, d = 1, 4, 8
        Q = np.random.randn(batch, n, d)
        K = np.random.randn(batch, n, d)
        V = np.random.randn(batch, n, d)
        out, attn = scaled_dot_product_attention(Q, K, V)
        self.assertEqual(out.shape, (batch, n, d))
        self.assertEqual(attn.shape, (batch, n, n))

    def test_attention_weights_sum_to_one_per_query(self):
        np.random.seed(0)
        Q = np.random.randn(1, 5, 8)
        K = np.random.randn(1, 5, 8)
        V = np.random.randn(1, 5, 8)
        _, attn = scaled_dot_product_attention(Q, K, V)
        self.assertTrue(np.allclose(attn.sum(axis=-1), 1.0))

    def test_attention_is_max_on_diagonal_when_Q_equals_K(self):
        # Si Q == K, cada token tiene mayor peso de atención sobre sí mismo
        Q = np.eye(4)[None, :, :]
        K = Q.copy()
        V = np.arange(16, dtype=float).reshape(1, 4, 4)
        _, attn = scaled_dot_product_attention(Q, K, V)
        for i in range(4):
            self.assertEqual(attn[0, i, i], attn[0, i].max())

    def test_causal_mask_blocks_future_tokens(self):
        # Con máscara causal, el token i no debe atender a tokens j>i
        np.random.seed(0)
        n = 4
        Q = np.random.randn(1, n, 8)
        K = np.random.randn(1, n, 8)
        V = np.random.randn(1, n, 8)
        # Máscara causal: True = permitido, False = bloqueado
        mask = np.tril(np.ones((n, n), dtype=bool))[None, :, :]
        _, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        # Atención sobre el futuro debe ser ~0
        for i in range(n):
            for j in range(i + 1, n):
                self.assertLess(attn[0, i, j], 1e-6)


if __name__ == "__main__":
    unittest.main()
