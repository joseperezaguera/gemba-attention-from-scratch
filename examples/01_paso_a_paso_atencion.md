# Atención paso a paso con "El gato come pescado"

> Walkthrough del demo [`demos/01_attention_basics.py`](../demos/01_attention_basics.py).

## Por qué este ejemplo importa

La atención escalar producto es la operación que define a un transformer. Aparece en cada capa, sobre cada token, en cada paso del entrenamiento y de la inferencia. Si entiendes esta única operación, entiendes el 80% de la mecánica que hace funcionar a GPT, Claude, Llama o Mistral. Antes de hablar de ventanas móviles o de attention sinks, conviene verla ejecutarse sobre una frase ridícula y corta — cuatro tokens, una matriz de 4×4 — para que no quede nada por imaginar.

## Cómo correrlo

```bash
uv run --with numpy --with matplotlib --python 3.11 python demos/01_attention_basics.py
```

O, si prefieres `pip`:

```bash
pip install numpy matplotlib
python3 demos/01_attention_basics.py
```

## Qué muestra

El script imprime por consola una tabla 4×4 con los pesos de atención entre los tokens `["El", "gato", "come", "pescado"]`. Cada fila representa un token-query, cada columna un token-key, y el valor es cuánto atiende el query al key. Las filas suman a 1, porque ese es el trabajo de la softmax.

Además, guarda un heatmap PNG en `demos/output/demo_01_attention_basics.png` con la misma información en color: amarillo = peso alto, morado = peso bajo.

Ejemplo de la tabla impresa:

```
                  El       gato       come    pescado
        El      0.250      0.250      0.250      0.250
      gato      ...
```

## Cómo interpretar la matriz

- **Filas:** quién pregunta. La fila de `gato` muestra a qué presta atención `gato` cuando construye su nueva representación.
- **Columnas:** quién es mirado. Si la columna `pescado` tiene valores altos, varios tokens están atendiendo mucho a `pescado`.
- **Diagonal:** cuánto se atiende cada token a sí mismo. En la atención vanilla con `Q=K=V=X`, los tokens tienden a parecerse a sí mismos (producto escalar máximo con uno mismo), pero la softmax reparte y suaviza.

Con embeddings aleatorios deterministas (`seed=42`) los pesos saldrán cercanos al uniforme (≈ 0.25 cada uno), porque no hay estructura semántica real. Eso es esperado y útil: prueba que la operación está bien implementada y que la softmax normaliza correctamente.

## Limitaciones

- Los embeddings son **ficticios y aleatorios deterministas**, no embeddings reales aprendidos por un modelo entrenado. Por eso los pesos no reflejan ninguna relación semántica entre las palabras españolas.
- Las proyecciones `Q`, `K`, `V` son la matriz `X` repetida tres veces (proyección identidad), no las multiplicaciones por matrices `W_Q`, `W_K`, `W_V` entrenadas. La operación matemática es idéntica; lo que falta es el aprendizaje.
- Es un único head, una única capa. Un transformer real apila docenas de capas con múltiples heads (ver [`src/multi_head.py`](../src/multi_head.py) para la versión multi-head).

Si quieres ver la atención con relaciones semánticas reales — el `gato` atendiendo de verdad a `come` y a `pescado` — necesitas tokens y embeddings de un modelo entrenado. Para eso, demo 05 carga Qwen2.5-3B-Instruct y mide el efecto sobre un texto largo.

## Vínculos

- Código del demo: [`demos/01_attention_basics.py`](../demos/01_attention_basics.py)
- Implementación de la operación: [`src/attention.py`](../src/attention.py)
- Visualización: [`src/visualize.py`](../src/visualize.py)
- Paper original: Vaswani et al. 2017, *Attention is All You Need*.
