"""Tests para attention sinks (StreamingLLM, Xiao et al. 2023)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from attention_sinks import attention_sinks_mask, attention_sinks_attention


class TestAttentionSinksMask(unittest.TestCase):
    def test_sinks_always_visible_for_far_token(self):
        """Con n=8, sinks=2, window=2, el token 7 ve {0,1} (sinks) + {6,7} (ventana)."""
        mask = attention_sinks_mask(n_tokens=8, sinks=2, window=2, causal=True)
        expected_t7 = np.array([1, 1, 0, 0, 0, 0, 1, 1], dtype=bool)
        np.testing.assert_array_equal(mask[7], expected_t7)

    def test_token_0_sees_only_itself(self):
        mask = attention_sinks_mask(n_tokens=8, sinks=2, window=2, causal=True)
        # Token 0 causal: solo se ve a sí mismo (la ventana también es 0..0)
        self.assertTrue(mask[0, 0])
        self.assertEqual(mask[0, 1:].sum(), 0)

    def test_sinks_0_equivalent_to_sliding_window(self):
        """Sin sinks (K=0), la máscara debe ser idéntica a sliding window pura."""
        from sliding_window import sliding_window_mask
        n = 8
        window = 3
        mask_sinks = attention_sinks_mask(n_tokens=n, sinks=0, window=window, causal=True)
        mask_sw = sliding_window_mask(n_tokens=n, window=window, causal=True)
        np.testing.assert_array_equal(mask_sinks, mask_sw)

    def test_middle_token_sees_sinks_and_window(self):
        """Token 5 con sinks=2 window=3: ve {0,1} + {3,4,5}."""
        mask = attention_sinks_mask(n_tokens=8, sinks=2, window=3, causal=True)
        expected_t5 = np.array([1, 1, 0, 1, 1, 1, 0, 0], dtype=bool)
        np.testing.assert_array_equal(mask[5], expected_t5)


class TestAttentionSinksAttention(unittest.TestCase):
    def test_output_shape(self):
        np.random.seed(0)
        batch, n, d = 1, 12, 4
        Q = np.random.randn(batch, n, d)
        K = np.random.randn(batch, n, d)
        V = np.random.randn(batch, n, d)
        out, attn = attention_sinks_attention(Q, K, V, sinks=2, window=4, causal=True)
        self.assertEqual(out.shape, (batch, n, d))
        self.assertEqual(attn.shape, (batch, n, n))

    def test_sink_tokens_receive_attention_from_far_tokens(self):
        """El token n-1 debe atender al sink (token 0)."""
        np.random.seed(0)
        batch, n, d = 1, 16, 4
        Q = np.random.randn(batch, n, d)
        K = np.random.randn(batch, n, d)
        V = np.random.randn(batch, n, d)
        _, attn = attention_sinks_attention(Q, K, V, sinks=2, window=4, causal=True)
        # El token 15 atiende al sink (token 0) con peso > 0
        self.assertGreater(attn[0, 15, 0], 0)

    def test_attention_zero_outside_sinks_and_window(self):
        """Token n-1 con sinks=2, window=3: NO debe ver tokens en {2,3,...,12}."""
        np.random.seed(0)
        batch, n, d = 1, 16, 4
        Q = np.random.randn(batch, n, d)
        K = np.random.randn(batch, n, d)
        V = np.random.randn(batch, n, d)
        _, attn = attention_sinks_attention(Q, K, V, sinks=2, window=3, causal=True)
        # Token 15 ve {0,1} + {13,14,15}
        for j in range(2, 13):
            self.assertLess(attn[0, 15, j], 1e-6, f"Token 15 no debe ver token {j}")


if __name__ == "__main__":
    unittest.main()
