---
description: Genera un heatmap del mapa de atención de una conversación reciente y muestra dónde podría haber lost-in-the-middle
---

Toma los últimos N tokens visibles de la conversación actual (o un texto proporcionado por el usuario), calcula el mapa de atención usando los módulos del repo `gemba-attention-from-scratch` (atención vanilla por defecto, o sliding window / attention sinks si el usuario lo pide), y guarda el heatmap como PNG.

Útil para entender visualmente qué partes del contexto un LLM podría haber ignorado.

Pasos:
1. Asegúrate de que el usuario tenga el repo clonado y `numpy + matplotlib` instalados (`pip install -r requirements.txt` desde el repo).
2. Carga `src/attention.py` y `src/visualize.py`.
3. Calcula `scaled_dot_product_attention` sobre embeddings ficticios deterministas para los tokens del usuario.
4. Llama a `plot_attention_heatmap` y guarda el PNG en `/tmp` o donde indique el usuario.
5. Devuelve la ruta del PNG y un análisis breve: qué tokens tienen pesos mayores, dónde está el "medio" más vulnerable.
