"""Tests para sliding window attention (Longformer, Mistral)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sliding_window import sliding_window_mask, sliding_window_attention


class TestSlidingWindowMask(unittest.TestCase):
    def test_causal_window_2_at_n_5(self):
        """Con n=5 y window=2 causal, cada token i ve [max(0, i-1), i]."""
        mask = sliding_window_mask(n_tokens=5, window=2, causal=True)
        expected = np.array([
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ], dtype=bool)
        np.testing.assert_array_equal(mask, expected)

    def test_window_size_1_is_diagonal(self):
        """window=1 causal: cada token solo se ve a sí mismo."""
        mask = sliding_window_mask(n_tokens=4, window=1, causal=True)
        np.testing.assert_array_equal(mask, np.eye(4, dtype=bool))

    def test_window_larger_than_n_equivalent_to_causal_full(self):
        """window=n causal equivale a la máscara causal completa."""
        n = 5
        mask = sliding_window_mask(n_tokens=n, window=n, causal=True)
        expected = np.tril(np.ones((n, n), dtype=bool))
        np.testing.assert_array_equal(mask, expected)

    def test_non_causal_bidirectional_window(self):
        """No causal: cada token i ve [i-w+1, i+w-1] simétrico."""
        mask = sliding_window_mask(n_tokens=5, window=2, causal=False)
        # Token 2 con window=2 no causal: ve {1, 2, 3}
        self.assertTrue(mask[2, 1])
        self.assertTrue(mask[2, 2])
        self.assertTrue(mask[2, 3])
        # No debe ver el 0 ni el 4
        self.assertFalse(mask[2, 0])
        self.assertFalse(mask[2, 4])


class TestSlidingWindowAttention(unittest.TestCase):
    def test_output_shape(self):
        np.random.seed(0)
        batch, n, d = 1, 8, 4
        Q = np.random.randn(batch, n, d)
        K = np.random.randn(batch, n, d)
        V = np.random.randn(batch, n, d)
        out, attn = sliding_window_attention(Q, K, V, window=3, causal=True)
        self.assertEqual(out.shape, (batch, n, d))
        self.assertEqual(attn.shape, (batch, n, n))

    def test_attention_outside_window_is_zero(self):
        """Token n-1 no debe atender al token 0 con window=3 causal."""
        np.random.seed(0)
        batch, n, d = 1, 8, 4
        Q = np.random.randn(batch, n, d)
        K = np.random.randn(batch, n, d)
        V = np.random.randn(batch, n, d)
        _, attn = sliding_window_attention(Q, K, V, window=3, causal=True)
        self.assertLess(attn[0, 7, 0], 1e-6)

    def test_attention_inside_window_is_nonzero(self):
        """Token n-1 sí debe atender a los W más recientes."""
        np.random.seed(0)
        batch, n, d = 1, 8, 4
        Q = np.random.randn(batch, n, d)
        K = np.random.randn(batch, n, d)
        V = np.random.randn(batch, n, d)
        _, attn = sliding_window_attention(Q, K, V, window=3, causal=True)
        # Los últimos 3 tokens (5, 6, 7) sí debe verlos
        self.assertGreater(attn[0, 7, 5], 0)
        self.assertGreater(attn[0, 7, 6], 0)
        self.assertGreater(attn[0, 7, 7], 0)


if __name__ == "__main__":
    unittest.main()
