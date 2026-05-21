"""
Demo 01: Atención básica con una frase ficticia.

Tokeniza manualmente "El gato come pescado" y muestra los pesos de atención
entre los cuatro tokens. Para ver, sin trucos, qué hace la operación más
básica del transformer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from attention import scaled_dot_product_attention
from visualize import plot_attention_heatmap


def main():
    tokens = ["El", "gato", "come", "pescado"]
    n = len(tokens)
    d = 16

    # Embeddings ficticios deterministas
    np.random.seed(42)
    X = np.random.randn(1, n, d)

    # Proyecciones identidad para simplicidad pedagógica
    Q, K, V = X, X, X
    out, attn = scaled_dot_product_attention(Q, K, V)

    print(f"Tokens: {tokens}")
    print(f"Pesos de atención (cada fila = cuánto atiende cada token a cada otro):")
    print()
    print(f"{'':>10} " + " ".join(f"{t:>10}" for t in tokens))
    for i, t in enumerate(tokens):
        row = " ".join(f"{attn[0, i, j]:>10.3f}" for j in range(n))
        print(f"{t:>10} {row}")

    output_path = Path(__file__).parent / "output" / "demo_01_attention_basics.png"
    plot_attention_heatmap(
        attn[0],
        labels=tokens,
        title="Atención vanilla — El gato come pescado",
        savepath=str(output_path),
    )
    print(f"\nHeatmap guardado en: {output_path}")


if __name__ == "__main__":
    main()
