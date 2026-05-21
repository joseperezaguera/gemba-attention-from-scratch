"""
Visualización de mapas de atención.

Convierte una matriz de pesos (n_queries × n_keys) en un heatmap PNG.
Útil para ver de un vistazo qué tokens está mirando cada token.
"""

from __future__ import annotations

import numpy as np


def plot_attention_heatmap(
    attn: np.ndarray,
    labels: list[str] | None = None,
    title: str | None = None,
    savepath: str | None = None,
    cmap: str = "viridis",
):
    """
    Dibuja un heatmap del mapa de atención.

    Args:
        attn: shape (n_queries, n_keys) — pesos de atención.
        labels: opcional, etiquetas para los ejes (tokens reales).
        title: opcional, título del gráfico.
        savepath: si se da, guarda el PNG. Si no, devuelve la figura.
        cmap: colormap de matplotlib.

    Returns:
        figure de matplotlib.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attn, cmap=cmap, aspect="auto", vmin=0, vmax=attn.max())
    fig.colorbar(im, ax=ax, label="Peso de atención")
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks(range(attn.shape[1]))
        ax.set_yticks(range(attn.shape[0]))
    if title:
        ax.set_title(title)
    ax.set_xlabel("Key positions")
    ax.set_ylabel("Query positions")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


if __name__ == "__main__":
    import tempfile
    from attention import scaled_dot_product_attention

    np.random.seed(0)
    n, d = 8, 4
    Q = np.random.randn(1, n, d)
    K = np.random.randn(1, n, d)
    V = np.random.randn(1, n, d)
    _, attn = scaled_dot_product_attention(Q, K, V)

    out = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    plot_attention_heatmap(attn[0], title="Atención vanilla, n=8", savepath=out)
    print(f"Heatmap guardado en: {out}")
