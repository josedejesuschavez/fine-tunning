"""Extrae los campos de un rate confirmation con el adaptador LoRA (requiere GPU).

Uso:
    uv run python src/fine_tunning/extract.py documento.txt
    cat documento.txt | uv run python src/fine_tunning/extract.py

Desde codigo:
    from fine_tunning.extract import Extractor
    extractor = Extractor()            # carga el modelo una sola vez
    datos = extractor.extract(texto)   # -> dict con los 6 campos
"""

import argparse
import contextlib
import json
import re
import sys
from datetime import date
from pathlib import Path

# Unsloth imprime sus banners en stdout; van a stderr para que stdout sea solo el JSON.
with contextlib.redirect_stdout(sys.stderr):
    from unsloth import FastLanguageModel

# La instruccion debe ser identica a la del entrenamiento.
from fine_tunning.generate_data import INSTRUCTION

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 160
FIELDS = ["broker", "load_number", "origin", "destination", "pickup_date", "rate"]
CITY_ST = re.compile(r"[^,]+, [A-Z]{2}")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ADAPTER = ROOT / "outputs" / "lora"


class ExtractionError(ValueError):
    """La respuesta del modelo no es un resultado valido; conserva la salida cruda."""

    def __init__(self, message, raw):
        super().__init__(message)
        self.raw = raw


def parse_output(raw):
    """Valida la respuesta del modelo. Rechaza en vez de adivinar: sin JSON limpio no hay resultado."""
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        raise ExtractionError("la respuesta no es JSON", raw) from None
    if not isinstance(data, dict) or set(data) != set(FIELDS):
        raise ExtractionError(f"se esperaban exactamente los campos {FIELDS}", raw)

    for f in ("broker", "load_number"):
        if not isinstance(data[f], str) or not data[f].strip():
            raise ExtractionError(f"{f} debe ser texto no vacio", raw)
    for f in ("origin", "destination"):
        if not isinstance(data[f], str) or not CITY_ST.fullmatch(data[f]):
            raise ExtractionError(f"{f} debe tener el formato 'City, ST'", raw)

    pickup = data["pickup_date"]
    if not isinstance(pickup, str) or not ISO_DATE.fullmatch(pickup):
        raise ExtractionError("pickup_date debe tener el formato YYYY-MM-DD", raw)
    try:
        date.fromisoformat(pickup)
    except ValueError:
        raise ExtractionError("pickup_date no es una fecha valida", raw) from None

    rate = data["rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        raise ExtractionError("rate debe ser un numero positivo", raw)
    return data


class Extractor:
    def __init__(self, adapter=DEFAULT_ADAPTER):
        with contextlib.redirect_stdout(sys.stderr):
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(adapter),
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)

    def generate(self, text):
        """Respuesta cruda del modelo, con el mismo prompt que en el entrenamiento."""
        if not text.strip():
            raise ValueError("el documento esta vacio")
        messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{text}"}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # El template ya incluye <|begin_of_text|>; sin esto el tokenizer agrega un segundo BOS.
        inputs = self.tokenizer(
            [prompt], return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        n_prompt = inputs["input_ids"].shape[1]
        if n_prompt + MAX_NEW_TOKENS > MAX_SEQ_LENGTH:
            raise ValueError(
                f"documento demasiado largo: {n_prompt} tokens de prompt "
                f"(maximo {MAX_SEQ_LENGTH - MAX_NEW_TOKENS})"
            )
        out = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True)

    def extract(self, text):
        """Dict con los 6 campos; lanza ExtractionError si la respuesta no es valida."""
        return parse_output(self.generate(text))


def main():
    parser = argparse.ArgumentParser(description="Extrae los campos de un rate confirmation.")
    parser.add_argument("file", nargs="?", help="archivo de texto; si se omite, lee de stdin")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="carpeta del adaptador LoRA")
    args = parser.parse_args()

    if args.file:
        try:
            text = Path(args.file).read_text(encoding="utf-8")
        except OSError as e:
            print(f"Error: no se pudo leer {args.file}: {e.strerror}", file=sys.stderr)
            sys.exit(1)
    else:
        text = sys.stdin.read()
    # Se valida antes de cargar el modelo, que tarda varios segundos.
    if not text.strip():
        print("Error: el documento esta vacio", file=sys.stderr)
        sys.exit(1)

    extractor = Extractor(args.adapter)
    try:
        data = extractor.extract(text)
    except (ExtractionError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        if isinstance(e, ExtractionError):
            print(f"Salida del modelo: {e.raw}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
