# Coste cuadrático: ver el O(n²) en una tabla

> Walkthrough del demo [`demos/02_quadratic_cost.py`](../demos/02_quadratic_cost.py).

## Por qué medir empíricamente

Que la atención escalar producto es O(n²) en tiempo y memoria es algo que aparece en todos los papers y en todos los blogs. Pero leer "O(n²)" es una cosa, y verlo es otra. Este demo mide tiempos reales con `time.perf_counter` y memoria aproximada sobre matrices `n × n` de pesos. Cuando dobles n y el tiempo se cuadruplique delante de tus ojos, la motivación detrás de sliding window, Flash Attention, y todo el zoo de variantes deja de ser abstracta. Por eso los modelos no pueden simplemente "tener más contexto" sin pagar por ello.

## Cómo correrlo

```bash
python demos/02_quadratic_cost.py
```

Solo necesita `numpy`. La memoria reportada es la matriz de scores `n × n` en `float32` (4 bytes por celda), que es donde el O(n²) duele primero.

## Qué ves

Una tabla con seis filas, una por cada tamaño de secuencia probado:

```
     n | tiempo (s) | mem aprox (MB) |  ratio
--------------------------------------------------
   100 |     0.0005 |           0.04 |  1.00×
   200 |     0.0018 |           0.15 |  3.60×
   500 |     0.0095 |           0.95 |  ...
  1000 |     ...
  2000 |     ...
  4000 |     ...
```

Los tiempos exactos varían con tu CPU, pero los ratios deberían contar la misma historia.

## Interpretación

- **Si duplicas n, el tiempo se multiplica aproximadamente por cuatro.** Es el sello distintivo de una operación cuadrática. Si fuera lineal, doblar n doblaría el tiempo (×2). Como es O(n²), doblar n cuadruplica (×4).
- **La memoria explota igual.** A `n = 4000` la matriz de scores ya ocupa ~60 MB para un solo head, una sola capa, un batch de 1. Un modelo real con 32 capas y 32 heads multiplicaría eso por mil sin pestañear.
- **Por debajo de n ≈ 200**, el ruido del sistema (caché, scheduler, BLAS) puede ensuciar las medidas y dar ratios anómalos. La tendencia se ve mejor en n grandes.

## Conexión con el artículo

Esta tabla es exactamente la que aparece en el §2 del artículo de Gemba [«¿Por qué los LLMs olvidan?»](https://www.gemba.es/). El cálculo está hecho con el mismo `numpy` que usaría cualquiera, sin trucos: la operación es genuinamente cuadrática, y por eso los proveedores cobran por tokens de entrada y salida, y por eso una conversación que no se trunca se vuelve carísima muy rápido.

## Por qué importa para la arquitectura

Si quisiéramos contextos de millones de tokens con atención completa, el coste sería prohibitivo: 1M tokens × 1M tokens × 4 bytes = 4 TB de memoria solo para la matriz de scores de una capa. De ahí salen las tres soluciones que recorre el resto del repo:

1. **Sliding window** ([`src/sliding_window.py`](../src/sliding_window.py)) — truncar a una ventana W: O(n·W).
2. **Attention sinks** ([`src/attention_sinks.py`](../src/attention_sinks.py)) — sliding window + anclas iniciales.
3. **Flash Attention** y kernels fusionados — no implementados aquí; reducen constantes pero la complejidad sigue siendo cuadrática.

## Vínculos

- Código del demo: [`demos/02_quadratic_cost.py`](../demos/02_quadratic_cost.py)
- Implementación medida: [`src/attention.py`](../src/attention.py)
