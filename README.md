# gemba-attention-from-scratch

> **Los mecanismos de atención que olvidan, implementados desde cero en Python sin librerías externas.** Material complementario del artículo de [Gemba](https://www.gemba.es/) [«¿Por qué los LLMs olvidan?»](https://www.gemba.es/) (25 de mayo de 2026).

Este repositorio no es código de producción. Es código **para entender**. Cada mecanismo de atención tiene su propio fichero, comentado paso a paso, con un `__main__` que se puede ejecutar para verlo en acción. Cero dependencias para la mecánica pura (solo `numpy`). Solo la demo viva del *lost in the middle* requiere `torch` + `transformers` para cargar GPT-2 small.

## Los tres mecanismos

| Mecanismo | Quién lo usa | Estrategia | Fichero |
|-----------|--------------|------------|---------|
| **Atención vanilla** (escalar producto) | El transformer original (Vaswani 2017) y todos sus descendientes | `softmax(Q·K^T/√d) · V` — cada token mira a todos los demás | [`src/attention.py`](src/attention.py) |
| **Sliding Window Attention** | Longformer (Beltagy 2020), Mistral 7B (Jiang 2023) | Truncar: cada token solo mira a los W más recientes | [`src/sliding_window.py`](src/sliding_window.py) |
| **Attention Sinks** | StreamingLLM (Xiao 2023) | Anclar los primeros K tokens siempre + ventana móvil | [`src/attention_sinks.py`](src/attention_sinks.py) |

## Empezar en 30 segundos

```bash
git clone https://github.com/joseperezaguera/gemba-attention-from-scratch.git
cd gemba-attention-from-scratch
pip install numpy matplotlib
python3 demos/01_attention_basics.py
python3 demos/02_quadratic_cost.py
python3 demos/03_sliding_vs_full.py
python3 demos/04_attention_sinks.py
```

Cada demo es autoejecutable y muestra un fenómeno concreto: la atención básica, el coste cuadrático con n, la diferencia entre atención completa y sliding window, y cómo los attention sinks evitan el colapso.

## Reproducir la curva en U con GPT-2

```bash
pip install torch transformers
python3 demos/05_lost_in_the_middle_gpt2.py
```

Inserta una "aguja" en distintas posiciones de un texto largo y mide cuánto la recuerda GPT-2 small. Reproduce, en pequeño, el fenómeno *lost in the middle* (Liu et al. 2024).

## Para entender vs para decidir

- Este repo: **para entender** la mecánica desde dentro.
- [`gemba-context-needle-runner`](https://github.com/joseperezaguera/gemba-context-needle-runner): **para decidir** midiendo el efecto en modelos comerciales (OpenAI, Anthropic) y abiertos (HuggingFace).

## Tests

```bash
python3 -m unittest discover tests
```

## Licencia

MIT.
