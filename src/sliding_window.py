"""
Sliding Window Attention — Longformer (Beltagy et al. 2020), Mistral 7B
(Jiang et al. 2023).

En lugar de que cada token mire a todos los demás (coste cuadrático en n),
cada token solo mira a una ventana móvil de los W más recientes. El coste
cae de O(n²) a O(n·W), lineal con n.

Lo que se pierde son las dependencias a larga distancia dentro de una sola
capa. En la práctica, al apilar muchas capas, el modelo accede a un contexto
efectivo de hasta capas × W tokens (en Mistral 7B: 32 × 4.096 = 131.072).
"""

from __future__ import annotations

import numpy as np

from attention import scaled_dot_product_attention


def sliding_window_mask(
    n_tokens: int,
    window: int,
    causal: bool = True,
) -> np.ndarray:
    """
    Devuelve máscara booleana (n_tokens, n_tokens) para sliding window.

    Args:
        n_tokens: longitud de la secuencia.
        window: tamaño de la ventana W. Cada token mira a W tokens.
        causal: si True, solo a los W anteriores (incluyéndose).
                Si False, ventana simétrica [i-w+1, i+w-1].

    Returns:
        mask[i, j] = True si el token i puede atender al token j.
    """
    mask = np.zeros((n_tokens, n_tokens), dtype=bool)
    for i in range(n_tokens):
        if causal:
            lo = max(0, i - window + 1)
            hi = i + 1
        else:
            lo = max(0, i - window + 1)
            hi = min(n_tokens, i + window)
        mask[i, lo:hi] = True
    return mask


def sliding_window_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    window: int,
    causal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Atención escalar producto restringida a una ventana móvil de W tokens.

    Args:
        Q: queries, shape (batch, n_tokens, d).
        K: keys, shape (batch, n_tokens, d).
        V: values, shape (batch, n_tokens, d_v).
        window: tamaño W de la ventana.
        causal: si True, ventana causal (solo pasado).

    Returns:
        out: (batch, n_tokens, d_v).
        attn: (batch, n_tokens, n_tokens) — pesos de atención (0 fuera de la
              ventana).
    """
    n_tokens = Q.shape[1]
    mask = sliding_window_mask(n_tokens, window, causal=causal)
    mask = mask[None, :, :]  # broadcast a (1, n_tokens, n_tokens)
    return scaled_dot_product_attention(Q, K, V, mask=mask)


if __name__ == "__main__":
    np.random.seed(0)
    n, d = 16, 8
    Q = np.random.randn(1, n, d)
    K = np.random.randn(1, n, d)
    V = np.random.randn(1, n, d)
    _, attn_full = scaled_dot_product_attention(Q, K, V)
    _, attn_sw = sliding_window_attention(Q, K, V, window=4, causal=True)
    print(f"Atención completa, fila del token {n-1} (todos los anteriores visibles):")
    print(f"  {attn_full[0, n-1]}")
    print(f"Sliding window (W=4), misma fila (solo 4 tokens visibles):")
    print(f"  {attn_sw[0, n-1]}")
