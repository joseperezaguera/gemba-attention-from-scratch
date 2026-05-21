"""
Demo 04: Cómo los attention sinks evitan el colapso.

Compara sliding window pura vs sliding window + attention sinks. Visualiza
cómo, con sinks, los primeros tokens siguen recibiendo atención aunque la
secuencia crezca mucho.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from sliding_window import sliding_window_attention
from attention_sinks import attention_sinks_attention


def main():
    np.random.seed(0)
    n = 48
    d = 16
    window = 6
    sinks = 4

    Q = np.random.randn(1, n, d)
    K = np.random.randn(1, n, d)
    V = np.random.randn(1, n, d)

    _, attn_sw = sliding_window_attention(Q, K, V, window=window, causal=True)
    _, attn_sinks = attention_sinks_attention(
        Q, K, V, sinks=sinks, window=window, causal=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, attn, title in zip(
        axes,
        [attn_sw[0], attn_sinks[0]],
        [f"Sliding window (W={window})", f"Sliding window + sinks (K={sinks}, W={window})"],
    ):
        im = ax.imshow(attn, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Key positions")
        ax.set_ylabel("Query positions")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    output_path = Path(__file__).parent / "output" / "demo_04_attention_sinks.png"
    fig.savefig(output_path, dpi=150)
    print(f"Comparación guardada en: {output_path}")
    print(f"En la imagen de la derecha, las primeras {sinks} columnas siguen activas aunque la ventana móvil haya pasado.")
    print(f"Esos son los attention sinks: anclas persistentes que evitan el colapso del modelo cuando la secuencia crece.")


if __name__ == "__main__":
    main()
