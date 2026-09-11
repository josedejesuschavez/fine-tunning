"""Mide que tan bien extrae el modelo los campos del rate confirmation.

Uso:
    uv run python src/fine_tunning/evaluate.py                  # modelo base
    uv run python src/fine_tunning/evaluate.py outputs/lora     # con adaptador LoRA
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
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_json(text):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def same(field, got, want):
    if got is None:
        return False
    if field == "rate":
        try:
            return abs(float(got) - float(want)) < 0.01
        except (TypeError, ValueError):
            return False
    return str(got).strip() == str(want).strip()


def main():
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    model, tokenizer = load_model(adapter)

    examples = [json.loads(line) for line in TEST_FILE.read_text(encoding="utf-8").splitlines()]

    valid_json = 0
    field_hits = {f: 0 for f in FIELDS}
    perfect = 0
    first_failure = None

    for i, ex in enumerate(examples, 1):
        raw = predict(model, tokenizer, ex)
        got = parse_json(raw)
        want = json.loads(ex["output"])

        if got is None:
            if first_failure is None:
                first_failure = (ex["input"], raw)
            print(f"[{i}/{len(examples)}] JSON invalido")
            continue

        valid_json += 1
        hits = 0
        for f in FIELDS:
            if same(f, got.get(f), want[f]):
                field_hits[f] += 1
                hits += 1
        if hits == len(FIELDS):
            perfect += 1
        elif first_failure is None:
            first_failure = (ex["input"], raw)
        print(f"[{i}/{len(examples)}] {hits}/{len(FIELDS)} campos")

    n = len(examples)
    print("\n" + "=" * 50)
    print(f"Modelo:            {adapter or BASE_MODEL}")
    print(f"JSON valido:       {valid_json}/{n}  ({100 * valid_json / n:.0f}%)")
    print(f"Extraccion exacta: {perfect}/{n}  ({100 * perfect / n:.0f}%)")
    print("\nPor campo:")
    for f in FIELDS:
        print(f"  {f:<15} {field_hits[f]:>3}/{n}  ({100 * field_hits[f] / n:.0f}%)")

    if first_failure:
        entrada, salida = first_failure
        print("\n--- Primer fallo ---")
        print("ENTRADA:", entrada[:300].replace("\n", " | "))
        print("SALIDA :", salida[:300])


if __name__ == "__main__":
    main()
