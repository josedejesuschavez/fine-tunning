"""Entrena un adaptador LoRA para extraer campos de rate confirmations.

Uso:
    uv run python src/fine_tunning/train.py
"""

import json
from dataclasses import fields
from pathlib import Path

from unsloth import FastLanguageModel  # debe importarse antes que trl/transformers

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_FILE = ROOT / "data" / "train.jsonl"
OUT_DIR = ROOT / "outputs" / "lora"


def build_dataset(tokenizer):
    rows = [json.loads(line) for line in TRAIN_FILE.read_text(encoding="utf-8").splitlines()]
    texts = []
    for r in rows:
        messages = [
            {"role": "user", "content": f"{r['instruction']}\n\n{r['input']}"},
            {"role": "assistant", "content": r["output"]},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return Dataset.from_dict({"text": texts})


def make_config():
    """Filtra los parametros segun lo que acepte la version instalada de trl."""
    wanted = dict(
        output_dir=str(ROOT / "outputs" / "checkpoints"),
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=5,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        bf16=True,
        seed=42,
        save_strategy="no",
        report_to="none",
    )
    valid = {f.name for f in fields(SFTConfig)}
    return SFTConfig(**{k: v for k, v in wanted.items() if k in valid})


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = build_dataset(tokenizer)
    print(f"\nEjemplos de entrenamiento: {len(dataset)}")
    print("--- muestra ---")
    print(dataset[0]["text"][:600])
    print("---------------\n")

    kwargs = dict(model=model, train_dataset=dataset, args=make_config())
    import inspect
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**kwargs)

    # Calcula el loss SOLO sobre la respuesta del assistant: enmascara el prompt
    # para que todo el gradiente se gaste en aprender a extraer, no a redactar.
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # Verifica el enmascarado: debe imprimir solo el JSON esperado.
    labels = trainer.train_dataset[0]["labels"]
    visible = [t for t in labels if t != -100]
    print("--- lo unico que el modelo aprende a generar ---")
    print(tokenizer.decode(visible))
    print("-----------------------------------------------\n")

    trainer.train()

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"\nAdaptador guardado en: {OUT_DIR}")


if __name__ == "__main__":
    main()
