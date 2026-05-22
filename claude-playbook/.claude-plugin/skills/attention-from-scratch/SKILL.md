---
name: attention-from-scratch
description: >
  Esta skill se activa cuando el usuario quiere experimentar con mecanismos
  de atención desde cero: atención escalar producto (Vaswani 2017), sliding
  window (Longformer, Mistral) o attention sinks (StreamingLLM). Útil para
  entender por qué los LLMs "olvidan" en el medio del contexto, qué cuesta
  la atención completa, cómo se trunca, y cómo evitar el colapso en
  streaming.

  Triggers: "explícame la atención", "por qué los LLMs olvidan en el medio",
  "qué es lost in the middle", "qué es sliding window attention",
  "cómo funcionan los attention sinks", "qué coste tiene la atención",
  "visualiza el mapa de atención", "qué pasa con la KV cache",
  "ventana de contexto", "por qué RAG no resuelve esto del todo".
version: 0.1.0
---

# Skill: Attention from Scratch

## Cuándo usar

Cuando el usuario quiera explorar la mecánica de atención que hay detrás de cualquier transformer moderno. Casos típicos:

- *"Explícame por qué los LLMs se olvidan de lo que hay en el medio del contexto."*
- *"¿Qué es exactamente la sliding window de Mistral?"*
- *"¿Cómo afectan los attention sinks a la generación en streaming?"*
- *"Visualiza el mapa de atención para esta frase."*
- *"¿Cuánto cuesta de verdad doblar la longitud del contexto?"*
- *"¿Por qué RAG no resuelve el problema del lost in the middle del todo?"*

## Cómo invocar el repo desde Claude Code

El repo es `gemba-attention-from-scratch`. Si el usuario no lo tiene clonado todavía:

```bash
git clone https://github.com/josemerca/gemba-attention-from-scratch.git
cd gemba-attention-from-scratch
pip install -r requirements.txt
```

Para la mecánica pura (demos 01-04), solo hace falta `numpy` y `matplotlib`. Para la demo viva del *lost in the middle* (demo 05), también `torch`, `transformers` y `accelerate` (descarga ~6GB de Qwen2.5-3B-Instruct la primera vez).

## Los cinco demos

| # | Demo | Qué muestra | Coste |
|---|------|-------------|-------|
| 01 | `demos/01_attention_basics.py` | Atención escalar producto sobre "El gato come pescado" — tabla de pesos + heatmap. | Instantáneo |
| 02 | `demos/02_quadratic_cost.py` | Mide tiempo y memoria al crecer n. Doblar n cuadruplica el coste. | Segundos |
| 03 | `demos/03_sliding_vs_full.py` | Compara atención completa vs sliding window — banda diagonal vs cuadrícula llena. | Instantáneo |
| 04 | `demos/04_attention_sinks.py` | Cómo los primeros K tokens fijos evitan el colapso de la sliding window pura. | Instantáneo |
| 05 | `demos/05_lost_in_the_middle.py` | Curva en U real con Qwen2.5-3B-Instruct sobre ~10k tokens. NLL más alto en el medio. | Varios minutos en CPU |

Cada demo es autoejecutable y deja un PNG o CSV en `demos/output/`.

## Los módulos `src/*.py`

| Fichero | Qué resuelve |
|---------|--------------|
| `src/attention.py` | Atención escalar producto: `softmax(Q·K^T/√d)·V`. La operación base. |
| `src/sliding_window.py` | Atención local con ventana W. Cada token mira solo a los W más recientes. |
| `src/attention_sinks.py` | Sliding window + K primeros tokens siempre visibles. Receta de StreamingLLM. |
| `src/multi_head.py` | Multi-head attention en NumPy puro. Para entender cómo se combinan H cabezas. |
| `src/positional.py` | Encodings sinusoidales (Vaswani) y RoPE simplificado. Sin esto, atención es ciega al orden. |
| `src/visualize.py` | `plot_attention_heatmap(...)`. Convierte una matriz (n × n) en PNG. |

## Cómo proceder ante una petición

1. **Identifica qué fenómeno** quiere entender el usuario. Si es "el LLM olvida" → demo 05 + sliding window + attention sinks. Si es "qué cuesta la atención" → demo 02. Si es "cómo se ve un mapa de atención" → demo 01 o el comando `/visualizar-atencion`.
2. **Mira si el ejemplo cabe** en los embeddings ficticios (rápido, pedagógico) o si necesita un modelo real (demo 05, mucho más lento).
3. **Ejecuta el demo apropiado** vía Bash, captura el output y, si genera un PNG, muéstraselo al usuario.
4. **Interpreta**: ata el resultado al fenómeno que el usuario preguntaba. Los heatmaps no se explican solos.
5. **Ofrece variaciones**: cambiar W, K, n, o el seed, según el caso.

## Comportamiento por defecto

- **Embeddings ficticios deterministas** (`seed=0` o `seed=42`) para todos los demos 01-04. Permiten reproducibilidad y no requieren GPU.
- **Para mapas sobre texto real**, usar embeddings ficticios derivados deterministamente del texto (hash → seed), o derivar al usuario al demo 05 si quiere atención real de un modelo entrenado.
- **Heatmaps siempre con `cmap='viridis'`** (consistente con `src/visualize.py`).
- **Decir siempre las limitaciones**: estos demos son pedagógicos. La atención real en GPT-4 o Claude usa Flash Attention, GQA, optimizaciones de kernel, y muchas más capas. La mecánica es la misma; los detalles de ingeniería no.

## Cosas que NO hace este repo

- **No carga modelos comerciales** (OpenAI, Anthropic). Para medir el efecto del lost-in-the-middle sobre modelos comerciales, redirigir al repo hermano `gemba-context-needle-runner`.
- **No implementa Flash Attention** ni kernels optimizados. Solo NumPy puro para que se vea la mecánica.
- **No reentrena modelos.** El demo 05 carga Qwen2.5-3B-Instruct preentrenado y mide perplejidad, no fine-tuning.

## Comandos disponibles

- `/visualizar-atencion` — Genera un heatmap del mapa de atención de un texto y comenta dónde podría haber lost-in-the-middle. Ver `commands/visualizar-atencion.md`.

## Punteros externos

- Artículo de Gemba [«¿Por qué los LLMs olvidan?»](https://www.gemba.es/) — 25 de mayo de 2026.
- Vaswani et al. 2017, *Attention is All You Need*.
- Liu et al. 2024, *Lost in the Middle: How Language Models Use Long Contexts*.
- Xiao et al. 2023, *Efficient Streaming Language Models with Attention Sinks*, [arXiv 2309.17453](https://arxiv.org/abs/2309.17453).
- Beltagy et al. 2020, *Longformer*.
- Jiang et al. 2023, *Mistral 7B*.

---

*Material complementario del artículo de Gemba [«¿Por qué los LLMs olvidan?»](https://www.gemba.es/) — 25 de mayo de 2026.*
