"""Mide que tan bien extrae el modelo los campos del rate confirmation.

Uso:
    uv run python src/fine_tunning/evaluate.py                  # modelo base
    uv run python src/fine_tunning/evaluate.py outputs/lora     # con adaptador LoRA

Los resultados se reportan por grupo del test (in_dist, nuevos_valores, nueva_plantilla).
"""

import json
import re
import sys
from pathlib import Path

from unsloth import FastLanguageModel

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024
FIELDS = ["broker", "load_number", "origin", "destination", "pickup_date", "rate"]

ROOT = Path(__file__).resolve().parent.parent.parent
TEST_FILE = ROOT / "data" / "test.jsonl"


def load_model(adapter=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter or BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def predict(model, tokenizer, example):
    messages = [{"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # El template ya incluye <|begin_of_text|>; sin esto el tokenizer agrega un segundo BOS.
    inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_json(text):
    """Primer objeto JSON del texto que tenga alguno de los campos esperados.

    raw_decode soporta objetos anidados (p. ej. origin como {"City": ..., "ST": ...}),
    que una regex no greedy cortaba en la primera llave de cierre.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and any(f in obj for f in FIELDS):
            return obj
    return None


def is_strict_json(text):
    """True si la respuesta es SOLO el JSON, sin texto ni ``` alrededor."""
    try:
        return isinstance(json.loads(text.strip()), dict)
    except json.JSONDecodeError:
        return False


def same(field, got, want):
    if got is None:
        return False
    if field == "rate":
        try:
            return abs(float(got) - float(want)) < 0.01
        except (TypeError, ValueError):
            return False
    return str(got).strip() == str(want).strip()


def new_stats():
    return {"n": 0, "valid": 0, "strict": 0, "perfect": 0,
            "fields": {f: 0 for f in FIELDS}, "first_failure": None}


def main():
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    model, tokenizer = load_model(adapter)

    examples = [json.loads(line) for line in TEST_FILE.read_text(encoding="utf-8").splitlines()]
    stats = {}

    for i, ex in enumerate(examples, 1):
        group = ex.get("group", "in_dist")
        s = stats.setdefault(group, new_stats())
        s["n"] += 1

        raw = predict(model, tokenizer, ex)
        got = parse_json(raw)
        want = json.loads(ex["output"])
        s["strict"] += is_strict_json(raw)

        hits = 0
        if got is not None:
            s["valid"] += 1
            for f in FIELDS:
                if same(f, got.get(f), want[f]):
                    s["fields"][f] += 1
                    hits += 1
        if hits == len(FIELDS):
            s["perfect"] += 1
        elif s["first_failure"] is None:
            s["first_failure"] = (ex["input"], ex["output"], raw)

        label = f"{hits}/{len(FIELDS)} campos" if got is not None else "JSON invalido"
        # flush=True: el avance se ve en vivo aunque la salida vaya a un archivo.
        print(f"[{i}/{len(examples)}] {group:<16} {label}", flush=True)

    total = new_stats()
    for s in stats.values():
        for key in ("n", "valid", "strict", "perfect"):
            total[key] += s[key]
        for f in FIELDS:
            total["fields"][f] += s["fields"][f]

    # json = hay un {...} parseable; estricto = la respuesta es SOLO el JSON.
    headers = ["json", "estricto", "exacto", "broker", "load", "origin", "dest", "fecha", "rate"]
    print("\n" + "=" * 101)
    print(f"Modelo: {adapter or BASE_MODEL}\n")
    print(f"{'grupo':<16}{'n':>4}" + "".join(f"{h:>9}" for h in headers))
    for g, s in [*stats.items(), ("TOTAL", total)]:
        values = [s["valid"], s["strict"], s["perfect"]] + [s["fields"][f] for f in FIELDS]
        print(f"{g:<16}{s['n']:>4}" + "".join(f"{100 * v / s['n']:>8.0f}%" for v in values))

    for g in stats:
        if stats[g]["first_failure"]:
            entrada, esperado, salida = stats[g]["first_failure"]
            print(f"\n--- Primer fallo [{g}] ---")
            print("ENTRADA :", entrada[:300].replace("\n", " | "))
            print("ESPERADO:", esperado)
            print("SALIDA  :", salida[:300].replace("\n", " | "))


if __name__ == "__main__":
    main()
