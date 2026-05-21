"""
Multi-Head Attention — Vaswani et al. 2017.

Divide el espacio d_model en n_heads cabezales, aplica atención escalar
producto en paralelo en cada cabezal con sus propias proyecciones Q/K/V,
concatena los resultados y proyecta de vuelta a d_model.

¿Por qué? Cada cabezal puede aprender a atender a un patrón distinto del
texto (sintaxis, semántica, dependencias largas, etc.). Es el equivalente
a tener varios lectores especializados en lugar de uno generalista.
"""

from __future__ import annotations

import numpy as np

from attention import scaled_dot_product_attention


def multi_head_attention(
    X: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    n_heads: int,
) -> np.ndarray:
    """
    Multi-head attention.

    Args:
        X: input, shape (batch, n_tokens, d_model).
        Wq, Wk, Wv: matrices de proyección Q/K/V, shape (d_model, d_model).
        Wo: matriz de proyección de salida, shape (d_model, d_model).
        n_heads: número de cabezales. Debe dividir a d_model.

    Returns:
        shape (batch, n_tokens, d_model).
    """
    batch, n_tokens, d_model = X.shape
    if d_model % n_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) debe ser divisible por n_heads ({n_heads})"
        )
    d_head = d_model // n_heads

    # Proyectar a Q, K, V
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Reorganizar para separar cabezales: (batch, n_heads, n_tokens, d_head)
    def split_heads(t: np.ndarray) -> np.ndarray:
        return t.reshape(batch, n_tokens, n_heads, d_head).transpose(0, 2, 1, 3)

    Q = split_heads(Q)
    K = split_heads(K)
    V = split_heads(V)

    # Atención por cabezal: aplanar batch×heads, llamar a SDPA, desaplanar
    Q_flat = Q.reshape(batch * n_heads, n_tokens, d_head)
    K_flat = K.reshape(batch * n_heads, n_tokens, d_head)
    V_flat = V.reshape(batch * n_heads, n_tokens, d_head)
    out_flat, _ = scaled_dot_product_attention(Q_flat, K_flat, V_flat)

    # Recomponer y proyectar
    out = out_flat.reshape(batch, n_heads, n_tokens, d_head)
    out = out.transpose(0, 2, 1, 3).reshape(batch, n_tokens, d_model)
    return out @ Wo


if __name__ == "__main__":
    np.random.seed(0)
    batch, n_tokens, d_model, n_heads = 1, 6, 16, 4
    X = np.random.randn(batch, n_tokens, d_model)
    Wq = np.random.randn(d_model, d_model) * 0.1
    Wk = np.random.randn(d_model, d_model) * 0.1
    Wv = np.random.randn(d_model, d_model) * 0.1
    Wo = np.random.randn(d_model, d_model) * 0.1
    out = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=n_heads)
    print(f"Multi-head: {n_heads} cabezales, d_model={d_model}, d_head={d_model // n_heads}")
    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {out.shape}")
