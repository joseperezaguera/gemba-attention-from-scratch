"""
Demo 05: Reproducir 'lost in the middle' con GPT-2 small.

Inserta una 'aguja' (una clave alfanumérica única dentro de una frase) en
distintas posiciones de un texto de relleno y mide la probabilidad
condicional que GPT-2 asigna a la respuesta correcta cuando se le pregunta
por la clave. Esperamos que la respuesta esté más fácil de predecir cuando
la aguja está cerca del principio o del final que cuando está en el medio.

Notas técnicas:
- GPT-2 small tiene ventana 1024 tokens. Es modelo antiguo y pequeño, así
  que la curva en U es más ruidosa que en modelos modernos como Claude o
  GPT-4. Para mediciones más serias, ver gemba-context-needle-runner.
- Requiere: torch, transformers, numpy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def insert_at_token_position(
    haystack_tokens: list[int],
    needle_tokens: list[int],
    pos_ratio: float,
) -> list[int]:
    """Inserta needle en la posición relativa pos_ratio del haystack."""
    n = len(haystack_tokens)
    insert_at = int(pos_ratio * n)
    return haystack_tokens[:insert_at] + needle_tokens + haystack_tokens[insert_at:]


def negative_log_likelihood(model, tokenizer, prompt_ids, answer_ids):
    """
    NLL promedio de los tokens de respuesta condicionados al prompt.
    Menor = mejor recuerdo.
    """
    full = torch.cat([prompt_ids, answer_ids], dim=1)
    # Solo computamos pérdida sobre los tokens de la respuesta
    labels = full.clone()
    labels[:, :prompt_ids.size(1)] = -100  # ignore_index
    with torch.no_grad():
        out = model(full, labels=labels)
    return out.loss.item()


def main():
    print("Cargando GPT-2 small (~500MB, la primera vez tarda)...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # Texto de relleno (haystack): párrafos genéricos
    haystack_text = (
        "Las arquitecturas de transformers han transformado el procesamiento "
        "del lenguaje natural en los últimos años. Los modelos preentrenados "
        "como BERT, GPT y T5 dominan benchmarks de comprensión y generación. " * 30
    )
    haystack_tokens = tokenizer.encode(haystack_text)
    # Limitamos a 700 tokens para que entren bien con la aguja y la pregunta
    haystack_tokens = haystack_tokens[:700]

    # Aguja única
    needle_text = " El código secreto del experimento es 7H9X. "
    needle_tokens = tokenizer.encode(needle_text)

    # Pregunta y respuesta esperada
    question = " ¿Cuál es el código secreto del experimento? El código es"
    question_ids = torch.tensor([tokenizer.encode(question)])
    answer = " 7H9X"
    answer_ids = torch.tensor([tokenizer.encode(answer)])

    positions = [0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95]

    print(f"\nPosición de la aguja → NLL del modelo sobre la respuesta esperada")
    print(f"(Menor NLL = mejor recuerdo)")
    print(f"{'pos':>6} | {'NLL':>10}")
    print("-" * 22)

    results = []
    for pos in positions:
        seq = insert_at_token_position(haystack_tokens, needle_tokens, pos)
        # Cortamos a la ventana máxima (1024) dejando sitio para pregunta y respuesta
        max_prompt = 1024 - question_ids.size(1) - answer_ids.size(1) - 10
        seq = seq[:max_prompt]
        prompt_ids = torch.tensor([seq])
        prompt_ids = torch.cat([prompt_ids, question_ids], dim=1)
        nll = negative_log_likelihood(model, tokenizer, prompt_ids, answer_ids)
        results.append((pos, nll))
        print(f"{pos:>6.2f} | {nll:>10.4f}")

    print()
    print("Si vemos la curva en U: NLL menor en pos=0.05 y pos=0.95, mayor en pos=0.5.")
    print("GPT-2 small es un modelo antiguo y pequeño; el efecto es ruidoso y a")
    print("veces no se ve nítido. Para mediciones serias contra modelos modernos,")
    print("usa gemba-context-needle-runner.")

    # Guardar resultados como CSV para post-análisis
    out_path = Path(__file__).parent / "output" / "demo_05_lost_in_the_middle_gpt2.csv"
    with open(out_path, "w") as f:
        f.write("position,nll\n")
        for pos, nll in results:
            f.write(f"{pos},{nll}\n")
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
