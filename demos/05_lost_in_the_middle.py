"""
Demo 05: Reproducir 'lost in the middle' con Phi-2 (microsoft/phi-2).

Inserta una 'aguja' única en distintas posiciones de un texto largo (heno)
y mide la perplejidad que el modelo asigna a la respuesta correcta cuando
se le pregunta por la clave. Esperamos curva en U: NLL bajo en posiciones
extremas (~0.05 y ~0.95) y alto en el medio (~0.5).

Phi-2 (2.7B, ventana 2.048 tokens) es lo bastante moderno como para que el
fenómeno se vea con más nitidez que con GPT-2 small. Para mediciones serias
contra modelos comerciales (Claude, GPT-4o), ver gemba-context-needle-runner.

Requiere: torch, transformers, numpy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/phi-2"
WINDOW = 2048
HAYSTACK_TOKENS = 1600  # dejar margen para la aguja + pregunta + respuesta


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
    print(f"Cargando {MODEL_ID} (~5GB, la primera vez tarda)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # float32 para CPU
        trust_remote_code=True,
    )
    model.eval()

    # Texto de relleno
    haystack_text = (
        "Las arquitecturas de transformers han transformado el procesamiento "
        "del lenguaje natural en los últimos años. Los modelos preentrenados "
        "como BERT, GPT y T5 dominan benchmarks de comprensión y generación. "
        "La atención escalar producto, propuesta en el artículo Attention is "
        "All You Need de 2017, es la operación fundamental sobre la que se "
        "construyen todos estos sistemas. " * 50
    )
    haystack_tokens = tokenizer.encode(haystack_text)
    haystack_tokens = haystack_tokens[:HAYSTACK_TOKENS]

    needle_text = " El código secreto del experimento es 7H9X. "
    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)

    question = " ¿Cuál es el código secreto del experimento? El código es"
    question_ids = torch.tensor([tokenizer.encode(question, add_special_tokens=False)])
    answer = " 7H9X"
    answer_ids = torch.tensor([tokenizer.encode(answer, add_special_tokens=False)])

    positions = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]

    print(f"\nModelo: {MODEL_ID} (ventana {WINDOW}, heno {HAYSTACK_TOKENS} tokens)")
    print(f"Posición de la aguja → NLL del modelo sobre la respuesta correcta")
    print(f"(Menor NLL = mejor recuerdo de la aguja)")
    print(f"{'pos':>6} | {'NLL':>10}")
    print("-" * 22)

    results = []
    for pos in positions:
        seq = insert_at_token_position(haystack_tokens, needle_tokens, pos)
        max_prompt = WINDOW - question_ids.size(1) - answer_ids.size(1) - 10
        seq = seq[:max_prompt]
        prompt_ids = torch.tensor([seq])
        prompt_ids = torch.cat([prompt_ids, question_ids], dim=1)
        nll = negative_log_likelihood(model, prompt_ids, answer_ids)
        results.append((pos, nll))
        print(f"{pos:>6.2f} | {nll:>10.4f}")

    # CSV de salida
    out_path = Path(__file__).parent / "output" / "demo_05_lost_in_the_middle.csv"
    with open(out_path, "w") as f:
        f.write("position,nll\n")
        for pos, nll in results:
            f.write(f"{pos},{nll}\n")

    # Análisis simple de curva en U
    min_pos = min(results, key=lambda x: x[1])
    max_pos = max(results, key=lambda x: x[1])
    print()
    print(f"NLL mínimo (mejor recuerdo): pos={min_pos[0]}, NLL={min_pos[1]:.4f}")
    print(f"NLL máximo (peor recuerdo):  pos={max_pos[0]}, NLL={max_pos[1]:.4f}")
    print()
    print(f"Curva en U esperada: NLL bajo en extremos (pos≈0.05 y pos≈0.95), alto en medio (pos≈0.5).")
    print(f"Resultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
