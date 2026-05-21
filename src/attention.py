"""
Atención escalar producto (Scaled Dot-Product Attention) — Vaswani et al. 2017.

La operación que define a un transformer: para cada token (query), calcular
cuánto le importa cada otro token (key) y, con esos pesos, mezclar sus
representaciones (values).

Fórmula:
    Atención(Q, K, V) = softmax(Q · K^T / √d) · V

Implementación pura en NumPy, sin librerías externas, para entender qué pasa
por dentro.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax numéricamente estable.

    Sustrae el máximo antes de exponenciar para evitar overflow con valores
    grandes. Matemáticamente idéntico a la softmax canónica.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Atención escalar producto: softmax(Q · K^T / √d) · V.

    Args:
        Q: queries, shape (batch, n_queries, d).
        K: keys, shape (batch, n_keys, d).
        V: values, shape (batch, n_keys, d_v).
        mask: opcional, shape broadcastable a (batch, n_queries, n_keys).
              True = posición visible, False = posición bloqueada.

    Returns:
        out:  shape (batch, n_queries, d_v) — la mezcla ponderada de values.
        attn: shape (batch, n_queries, n_keys) — los pesos de atención.
    """
    d = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    attn = softmax(scores, axis=-1)
    out = np.matmul(attn, V)
    return out, attn


if __name__ == "__main__":
    # Demo mínima: tokenización ficticia de "El gato come pescado"
    np.random.seed(0)
    n_tokens = 4
    d = 8
    Q = np.random.randn(1, n_tokens, d)
    K = np.random.randn(1, n_tokens, d)
    V = np.random.randn(1, n_tokens, d)
    out, attn = scaled_dot_product_attention(Q, K, V)
    print(f"Tokens: {n_tokens}, dimensión: {d}")
    print(f"Pesos de atención (n_queries × n_keys):\n{attn[0]}")
    print(f"Cada fila suma a {attn[0].sum(axis=-1)}")
