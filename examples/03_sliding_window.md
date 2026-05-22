# Sliding window: atención local que escala

> Walkthrough del demo [`demos/03_sliding_vs_full.py`](../demos/03_sliding_vs_full.py).

## Por qué la sliding window

La atención escalar producto cuesta O(n²). Si lo único que necesitas, en la mayoría de capas, son las dependencias locales (las relaciones entre palabras cercanas dentro de la misma frase o párrafo), pagar por mirar a todos los demás tokens es un derroche. Longformer (Beltagy et al. 2020) y Mistral 7B (Jiang et al. 2023) optaron por la solución más obvia: que cada token solo mire a una ventana W de los más recientes. El coste cae de O(n²) a O(n·W), lineal con n cuando W queda fija.

Lo que se pierde en una sola capa son las dependencias a larga distancia. Lo que se gana, al apilar muchas capas, es un contexto efectivo amplio: en Mistral 7B, 32 capas × 4 096 de ventana = 131 072 tokens visibles a través de la pila, sin pagar el O(n²) en ninguna capa individual.

## Cómo correrlo

```bash
python demos/03_sliding_vs_full.py
```

Solo necesita `numpy` y `matplotlib`. Genera un PNG con dos heatmaps lado a lado en `demos/output/demo_03_sliding_vs_full.png`.

## Qué ves

Dos mapas de atención sobre la misma secuencia de n=32 tokens (embeddings aleatorios deterministas con `seed=0`):

- **Izquierda — Atención completa.** Toda la cuadrícula 32×32 está coloreada. Cada token-query (eje vertical) atiende a cada token-key (eje horizontal). Los pesos son pequeños y dispersos porque la softmax los reparte entre 32 candidatos.
- **Derecha — Sliding window W=6, causal.** Solo se ven valores no nulos en una banda diagonal de ancho 6. Fuera de la banda, todo es cero: la máscara ha bloqueado esas posiciones antes de aplicar la softmax.

La banda diagonal es el rasgo visual característico de la sliding window. Cuanto más estrecha la banda, más local la atención. Cuanto más ancha, más se parece a la atención completa.

## Cuándo conviene

- **Documentos largos** donde la mayoría de relaciones útiles son locales (texto continuo, código, transcripciones).
- **Generación en streaming**, donde el KV cache crecería sin límite con atención completa. Con sliding window W queda acotado a W entradas por capa.
- **Modelos que necesitan escalar a contextos de 100k+ tokens** sin coste cuadrático prohibitivo.
- **Capas inferiores** de un modelo, donde los patrones suelen ser locales (fonología, morfología, sintaxis cercana). Algunas arquitecturas mezclan sliding window en capas bajas y atención completa en capas altas.

## Cuándo no conviene

- **Dependencias muy a larga distancia dentro de una sola capa.** Si necesitas que el token 5 000 condicione directamente al token 5 mediante una única operación de atención, sliding window con W=512 no va a verlo. Tendrás que confiar en que la pila de capas propague la información — y eso no siempre funciona bien.
- **Tareas tipo *needle in a haystack*** muy adversariales, donde una pista pequeña en una posición arbitraria del contexto debe rescatarse al final. La sliding window pura es especialmente vulnerable aquí; combinarla con attention sinks (demo 04) mitiga parte del problema.
- **Cuando el coste cuadrático no es el cuello de botella.** Si tu contexto cabe holgadamente en memoria con atención completa, complicar la implementación con máscaras no aporta nada.

## Detalles que es fácil pasar por alto

- **Causalidad.** En modo `causal=True` (el que usa el demo), un token solo mira hacia atrás. Eso replica la generación autoregresiva. Sin causalidad, la ventana es simétrica `[i-W+1, i+W-1]`.
- **La softmax solo se normaliza sobre las posiciones visibles**, no sobre todas. Por eso los valores no nulos de la banda son mayores que en la atención completa equivalente — están repartiendo entre 6 candidatos, no entre 32.
- **El tamaño efectivo de la ventana depende del modelo**: Mistral 7B usa 4 096, Longformer original 512. No hay un valor canónico; es un trade-off entre coste y calidad.

## Vínculos

- Código del demo: [`demos/03_sliding_vs_full.py`](../demos/03_sliding_vs_full.py)
- Implementación: [`src/sliding_window.py`](../src/sliding_window.py)
- Atención vanilla para comparar: [`src/attention.py`](../src/attention.py)
- Papers: Beltagy et al. 2020 (*Longformer*); Jiang et al. 2023 (*Mistral 7B*).
