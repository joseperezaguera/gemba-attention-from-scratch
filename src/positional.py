"""
Embeddings posicionales: sinusoidal (Vaswani et al. 2017) y RoPE
(Su et al. 2021, Rotary Position Embeddings).

Los transformers procesan tokens en paralelo y, sin información adicional,
serían invariantes al orden. Los embeddings posicionales codifican la
posición de cada token para que la atención pueda usar esa información.

- Sinusoidal: añade un vector fijo a cada token según su posición.
- RoPE: rota cada token en el plano complejo según su posición. Preserva
  la norma. Es el estándar actual en Llama, GPT-NeoX, Mistral y similares.
"""

from __future__ import annotations

import numpy as np


def sinusoidal_encoding(n_tokens: int, d_model: int) -> np.ndarray:
    """
    Encoding posicional sinusoidal (Vaswani et al. 2017).

    Para cada posición pos y cada dimensión i:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Returns:
        shape (n_tokens, d_model), valores en [-1, 1].
    """
    enc = np.zeros((n_tokens, d_model))
    pos = np.arange(n_tokens)[:, None].astype(float)
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    enc[:, 0::2] = np.sin(angle[:, 0::2])
    enc[:, 1::2] = np.cos(angle[:, 1::2])
    return enc


def rope_apply(x: np.ndarray) -> np.ndarray:
    """
    Rotary Position Embeddings (Su et al. 2021) — versión simplificada.

    Rota pares de dimensiones del vector según la posición del token. Como
    es una rotación, preserva la norma euclídea. La frecuencia de rotación
    decae con la dimensión (más rápido en las primeras, más lento en las
    últimas), igual que en sinusoidal.

    Args:
        x: shape (..., n_tokens, d_model) con d_model par.

    Returns:
        Mismo shape, con rotación aplicada.
    """
    *batch, n_tokens, d = x.shape
    if d % 2 != 0:
        raise ValueError("d_model debe ser par para RoPE")
    half = d // 2
    pos = np.arange(n_tokens)[:, None]
    freqs = 1.0 / (10000.0 ** (np.arange(half) / half))
    theta = pos * freqs[None, :]
    cos = np.cos(theta)
    sin = np.sin(theta)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    out = np.empty_like(x)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


if __name__ == "__main__":
    print("Sinusoidal encoding (n=8, d=4):")
    print(sinusoidal_encoding(8, 4))
    print()
    np.random.seed(0)
    x = np.random.randn(1, 5, 8)
    out = rope_apply(x)
    print(f"RoPE preserva norma: input={np.linalg.norm(x, axis=-1)}, output={np.linalg.norm(out, axis=-1)}")
