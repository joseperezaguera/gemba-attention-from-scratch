"""
Demo 05: Reproducir 'lost in the middle' con Qwen2.5-3B-Instruct.

Inserta una 'aguja' única en distintas posiciones de un texto largo (~10k
tokens) y mide la perplejidad que el modelo asigna a la respuesta correcta
cuando se le pregunta por la clave. Esperamos curva en U: NLL bajo en
posiciones extremas (~0.05 y ~0.95) y alto en el medio (~0.5).

Por qué Qwen2.5-3B-Instruct: ventana de 128k tokens (suficiente para
contextos largos), modelo abierto reciente y robusto, ~6GB de descarga.
El fenómeno lost-in-the-middle se ve nítido con contextos largos
(10k+ tokens); con contextos cortos (1-2k) suele no aparecer.

Requiere: torch, transformers, accelerate, numpy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
HAYSTACK_TOKENS = 10000  # ~10k tokens de heno: medio realmente lejano de los extremos


def insert_at_token_position(
    haystack_tokens: list[int],
    needle_tokens: list[int],
    pos_ratio: float,
) -> list[int]:
    """Inserta needle en la posición relativa pos_ratio del haystack."""
    n = len(haystack_tokens)
    insert_at = int(pos_ratio * n)
    return haystack_tokens[:insert_at] + needle_tokens + haystack_tokens[insert_at:]


def negative_log_likelihood(model, prompt_ids, answer_ids):
    """NLL promedio de los tokens de respuesta condicionados al prompt."""
    full = torch.cat([prompt_ids, answer_ids], dim=1)
    labels = full.clone()
    labels[:, :prompt_ids.size(1)] = -100
    with torch.no_grad():
        out = model(full, labels=labels)
    return out.loss.item()


def main():
    print(f"Cargando {MODEL_ID} (~6GB, la primera vez tarda)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # float32 para CPU
        trust_remote_code=True,
    )
    model.eval()

    # Texto de relleno diverso (no solo repetición): tres temas distintos
    # para evitar artefactos de tokenización por repetición exacta.
    base_paragraphs = [
        # Tema 1: ML/LLMs
        "Las arquitecturas de transformers han transformado el procesamiento "
        "del lenguaje natural. Los modelos preentrenados como BERT, GPT y T5 "
        "dominan los benchmarks de comprensión y generación. La atención "
        "escalar producto, propuesta en el paper Attention is All You Need "
        "de 2017, es la operación fundamental. ",
        # Tema 2: biología
        "La fotosíntesis es el proceso por el cual las plantas convierten "
        "energía lumínica en energía química, almacenada en moléculas de "
        "glucosa. Ocurre principalmente en los cloroplastos, en presencia "
        "de clorofila. El oxígeno se libera como subproducto. ",
        # Tema 3: historia
        "El Renacimiento fue un movimiento cultural surgido en Italia en "
        "el siglo XIV, caracterizado por un renovado interés en la "
        "antigüedad clásica. Figuras como Leonardo da Vinci, Miguel Ángel "
        "y Rafael definieron el arte de la época. ",
    ]
    # Construye un heno variado intercalando los párrafos
    haystack_text = "".join(base_paragraphs * 200)
    haystack_tokens = tokenizer.encode(haystack_text)[:HAYSTACK_TOKENS]
    print(f"Heno: {len(haystack_tokens)} tokens.")

    needle_text = " El código secreto del experimento es 7H9X. "
    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)

    question = " ¿Cuál es el código secreto del experimento? El código es"
    question_ids = torch.tensor([tokenizer.encode(question, add_special_tokens=False)])
    answer = " 7H9X"
    answer_ids = torch.tensor([tokenizer.encode(answer, add_special_tokens=False)])

    positions = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]

    print(f"\nModelo: {MODEL_ID}")
    print(f"Heno: {HAYSTACK_TOKENS} tokens, aguja en distintas posiciones, pregunta al final.")
    print(f"\n{'pos':>6} | {'NLL':>10}")
    print("-" * 22)

    results = []
    for pos in positions:
        seq = insert_at_token_position(haystack_tokens, needle_tokens, pos)
        prompt_ids = torch.tensor([seq])
        prompt_ids = torch.cat([prompt_ids, question_ids], dim=1)
        nll = negative_log_likelihood(model, prompt_ids, answer_ids)
        results.append((pos, nll))
        print(f"{pos:>6.2f} | {nll:>10.4f}")

    # CSV
    out_path = Path(__file__).parent / "output" / "demo_05_lost_in_the_middle.csv"
    with open(out_path, "w") as f:
        f.write("position,nll\n")
        for pos, nll in results:
            f.write(f"{pos},{nll}\n")

    # Análisis
    min_pos = min(results, key=lambda x: x[1])
    max_pos = max(results, key=lambda x: x[1])
    print()
    print(f"NLL mínimo (mejor recuerdo): pos={min_pos[0]}, NLL={min_pos[1]:.4f}")
    print(f"NLL máximo (peor recuerdo):  pos={max_pos[0]}, NLL={max_pos[1]:.4f}")
    print(f"\nResultados guardados en: {out_path}")
    print()
    print("Curva en U esperada: NLL bajo en extremos (pos≈0.05 y pos≈0.95), alto en medio (pos≈0.5).")
    print("Si NO se ve, considerar contexto aún más largo (32k+ tokens) o modelo más grande.")


if __name__ == "__main__":
    main()
