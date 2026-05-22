# Attention sinks: por qué los primeros tokens nunca se van

> Walkthrough del demo [`demos/04_attention_sinks.py`](../demos/04_attention_sinks.py).

## El problema que resuelven

La sliding window funciona bien hasta que no funciona. En generación en streaming — es decir, cuando el modelo emite tokens uno detrás de otro y la secuencia crece más allá de la ventana W — los primeros tokens del contexto acaban saliendo por el borde. Y, en cuanto eso ocurre, muchos modelos colapsan: la perplejidad se dispara, las salidas se vuelven incoherentes, el texto generado degenera.

Esto no es una anécdota: lo midieron Xiao et al. en *StreamingLLM* (2023, [arXiv 2309.17453](https://arxiv.org/abs/2309.17453)) sobre Llama-2, MPT, Falcon y Pythia. Todos exhibían el mismo patrón. Y la razón resultó ser tan elegante como inesperada.

## Por qué los attention sinks existen

Cuando una capa de atención calcula la softmax, los pesos están obligados a sumar uno. Eso es matemática, no diseño. Si un token-query no encuentra a quién atender bien — porque el contexto local no le aporta información útil — la softmax sigue exigiendo que reparta uno entero en algún sitio. Y "algún sitio", durante el entrenamiento, tiende a ser los primeros tokens de la secuencia: los que siempre están ahí, independientemente del prompt.

Los modelos aprenden a **depositar atención sobrante en los primeros K tokens**, da igual que su contenido sea relevante (un `<bos>`, un espacio, basura). Se convierten en *attention sinks*: receptores de la atención que sobra.

Cuando una sliding window pura empuja esos tokens fuera de la ventana, los sinks desaparecen y el sistema se desestabiliza. La solución de StreamingLLM es trivial: **conservar siempre los primeros K tokens, además de la ventana móvil normal**. Con K=4 ya es suficiente. Con eso, modelos sin reentrenar consiguen generar de forma estable hasta cuatro millones de tokens.

## Cómo correrlo

```bash
python demos/04_attention_sinks.py
```

Solo `numpy` y `matplotlib`. Genera un PNG con dos heatmaps lado a lado en `demos/output/demo_04_attention_sinks.png`.

## Qué ves

Dos mapas de atención sobre la misma secuencia de n=48 tokens, todos en modo causal:

- **Izquierda — Sliding window W=6.** La banda diagonal característica. Cuando el query está en la posición 40, atiende a los keys 35-40. Los keys 0-3 quedaron fuera hace mucho.
- **Derecha — Sliding window W=6 + sinks K=4.** La banda diagonal sigue ahí, **y además las cuatro primeras columnas están iluminadas para todas las filas**. Los tokens 0, 1, 2 y 3 reciben peso aunque el query esté en la posición 47.

Esa segunda imagen es la firma visual de StreamingLLM. Un L invertida: la banda de la sliding window más la franja vertical persistente de los sinks.

## Por qué funciona

Los sinks dan a la softmax un lugar estable donde depositar atención sobrante. El modelo no tiene que improvisar candidatos cuando el contexto local no le sirve; ya hay un destino fijo. Y como el modelo ya aprendió durante el entrenamiento a usar los primeros tokens así, no necesita fine-tuning para que la cosa funcione: se aprovecha de un comportamiento que ya tenía.

Es un caso bonito de ingeniería derivada de observar de cerca lo que un sistema hace. No es una mejora arquitectónica grande; es entender un detalle de la softmax y dejarlo en su sitio.

## Cuándo importa

- **Inferencia con contextos que crecen sin límite** (chats largos, asistentes con memoria persistente, generación de documentos extensos).
- **Modelos pequeños o medianos en streaming** donde reentrenar con contexto largo es caro.
- **Cualquier caso donde sliding window sola degrada con secuencias largas** y el coste de añadir K tokens fijos es trivial.

## Cuándo no aporta

- **Si el contexto cabe en la ventana** y nunca se desborda, los sinks no se ejercitan: estarías pagando 4 entradas de KV cache que no usas.
- **Atención completa sin truncar** no tiene el problema porque los tokens iniciales no se pierden nunca.

## Vínculos

- Código del demo: [`demos/04_attention_sinks.py`](../demos/04_attention_sinks.py)
- Implementación: [`src/attention_sinks.py`](../src/attention_sinks.py)
- Sliding window para comparar: [`src/sliding_window.py`](../src/sliding_window.py)
- Paper: Xiao et al. 2023, *Efficient Streaming Language Models with Attention Sinks*, [arXiv 2309.17453](https://arxiv.org/abs/2309.17453).
