"""
Attention Sinks — StreamingLLM (Xiao et al. 2023, arXiv 2309.17453).

La sliding window pura tiene un problema raro y persistente: cuando los
primeros tokens de la secuencia llegan al borde de la ventana y desaparecen,
los modelos empiezan a colapsar. Salidas incoherentes, perplejidad disparada.

Xiao y su equipo descubrieron la razón: los primeros tokens, da igual su
contenido (basura, espacios, `<bos>`), acumulan una cantidad enorme de
atención durante el entrenamiento. La softmax obliga a que los pesos sumen
uno y, cuando un token no encuentra a quién atender bien, "deposita" peso
en cualquier sitio. Los primeros tokens, por construcción, siempre están
ahí. Se vuelven *attention sinks*.

La solución de StreamingLLM: preservar los primeros K tokens del contexto
SIEMPRE, además de la sliding window normal. Con esa combinación, un modelo
puede generar de forma estable hasta cuatro millones de tokens sin
fine-tuning, sin colapsar.
"""

from __future__ import annotations

import numpy as np

from attention import scaled_dot_product_attention


def attention_sinks_mask(
    n_tokens: int,
    sinks: int,
    window: int,
    causal: bool = True,
) -> np.ndarray:
    """
    Máscara estilo StreamingLLM: anclas iniciales + ventana móvil.

    Args:
        n_tokens: longitud de la secuencia.
        sinks: número K de tokens iniciales siempre visibles.
        window: tamaño W de la ventana móvil.
        causal: si True, ventana causal.

    Returns:
        mask[i, j] = True si el token i puede atender al token j.
    """
    mask = np.zeros((n_tokens, n_tokens), dtype=bool)
    for i in range(n_tokens):
        # Sinks: tokens 0..K-1 siempre visibles
        # En modo causal, solo si ya han aparecido (j <= i).
        sink_end = min(sinks, i + 1) if causal else sinks
        mask[i, :sink_end] = True
        # Ventana móvil
        if causal:
            lo = max(sinks, i - window + 1) if i >= sinks else i
            hi = i + 1
        else:
            lo = max(sinks, i - window + 1)
            hi = min(n_tokens, i + window)
        mask[i, lo:hi] = True
    return mask


def attention_sinks_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    sinks: int,
    window: int,
    causal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Atención escalar producto con attention sinks + sliding window.

    Args:
        Q, K, V: (batch, n_tokens, d).
        sinks: K tokens iniciales siempre visibles.
        window: ventana móvil de los últimos W tokens.

    Returns:
        out, attn — como scaled_dot_product_attention.
    """
    n_tokens = Q.shape[1]
    mask = attention_sinks_mask(n_tokens, sinks, window, causal=causal)
    mask = mask[None, :, :]
    return scaled_dot_product_attention(Q, K, V, mask=mask)


if __name__ == "__main__":
    np.random.seed(0)
    n, d = 16, 8
    Q = np.random.randn(1, n, d)
    K = np.random.randn(1, n, d)
    V = np.random.randn(1, n, d)
    _, attn = attention_sinks_attention(Q, K, V, sinks=2, window=4, causal=True)
    print(f"Attention sinks, sinks=2, window=4, n={n}")
    print(f"Token {n-1} atiende a:")
    for j in range(n):
        if attn[0, n-1, j] > 1e-6:
            print(f"  token {j}: peso {attn[0, n-1, j]:.4f}")
