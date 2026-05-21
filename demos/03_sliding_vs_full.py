"""
Demo 03: Atención completa vs sliding window.

Compara, sobre la misma secuencia, los mapas de atención de la versión
vanilla (cada token mira a todos) y la versión sliding window de
Mistral/Longformer (cada token solo mira a una ventana móvil). Guarda
ambos heatmaps lado a lado.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from attention import scaled_dot_product_attention
from sliding_window import sliding_window_attention


def main():
    np.random.seed(0)
    n = 32
    d = 16
    window = 6

    Q = np.random.randn(1, n, d)
    K = np.random.randn(1, n, d)
    V = np.random.randn(1, n, d)

    _, attn_full = scaled_dot_product_attention(Q, K, V)
    _, attn_sw = sliding_window_attention(Q, K, V, window=window, causal=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, attn, title in zip(
        axes,
        [attn_full[0], attn_sw[0]],
        [f"Atención completa (n={n})", f"Sliding window W={window} (causal)"],
    ):
        im = ax.imshow(attn, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Key positions")
        ax.set_ylabel("Query positions")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    output_path = Path(__file__).parent / "output" / "demo_03_sliding_vs_full.png"
    fig.savefig(output_path, dpi=150)
    print(f"Comparación guardada en: {output_path}")
    print(f"En la imagen de la derecha, los ceros fuera de la diagonal son las posiciones que la ventana móvil ha truncado.")


if __name__ == "__main__":
    main()
