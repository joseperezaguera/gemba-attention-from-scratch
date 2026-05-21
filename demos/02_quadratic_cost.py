"""
Demo 02: El coste cuadrático.

Mide cuánto tarda la atención escalar producto cuando crece n. La
multiplicación de Q por K^T es O(n²) en tiempo y memoria. Duplicar n
debería cuadruplicar el tiempo.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from attention import scaled_dot_product_attention


def main():
    np.random.seed(0)
    d = 64
    print(f"Atención escalar producto con d={d}.")
    print(f"{'n':>6} | {'tiempo (s)':>10} | {'mem aprox (MB)':>14} | {'ratio':>6}")
    print("-" * 50)

    base_t = None
    for n in [100, 200, 500, 1000, 2000, 4000]:
        Q = np.random.randn(1, n, d).astype(np.float32)
        K = Q.copy()
        V = Q.copy()
        # warm-up
        scaled_dot_product_attention(Q, K, V)
        t0 = time.perf_counter()
        scaled_dot_product_attention(Q, K, V)
        t = time.perf_counter() - t0
        if base_t is None:
            base_t = t
        # Memoria aprox: matriz scores (n x n) en float32 = 4 bytes
        mem_mb = (n * n * 4) / (1024 ** 2)
        print(f"{n:>6} | {t:>10.4f} | {mem_mb:>14.2f} | {t/base_t:>6.2f}×")

    print()
    print("Si la implementación fuera lineal con n, el ratio crecería como x2 al doblar n.")
    print("Como es cuadrática, el ratio crece aproximadamente como x4. La memoria explota igual.")


if __name__ == "__main__":
    main()
