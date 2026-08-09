"""
2_train_qlora.py
Fine-tuning QLoRA sobre train.jsonl/val.jsonl generados por 2_generar_dataset_llm.py.

Cada ejemplo es {"messages": [...], "esi_level": N} con roles system/user/assistant,
igual que se generó offline con LLM+RAG. Este script NO usa RAG — el modelo
aprende el comportamiento (0/1/2 preguntas, bandera roja, formato "Triage: N - ...")
directamente de los datos.

La pérdida (loss) se calcula SOLO sobre los tokens de los turnos "assistant"
(preguntas de la enfermera y clasificación final) — los turnos "system" y "user"
se enmascaran (label = -100) para que el modelo no aprenda a "predecir" lo que
dice el paciente, solo a responder como enfermera.

Uso:
  python 2_train_qlora.py \
      --train data_v2/train.jsonl \
      --val data_v2/val.jsonl \
      --base-model Qwen/Qwen2.5-3B-Instruct \
      --out trained_model/final_model \
      --epochs 3
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                           Trainer, TrainingArguments)

IGNORE_INDEX = -100


def cargar_jsonl(path):
    ejemplos = []
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                ejemplos.append(json.loads(linea))
    return ejemplos


def tokenizar_con_mascara(ejemplo, tok, max_len):
    """Tokeniza la conversación completa y marca con -100 todo lo que NO sea
    un turno 'assistant', para que el loss solo se calcule sobre lo que el
    modelo debe aprender a generar (preguntas de enfermera + clasificación).

    Nota de compatibilidad: apply_chat_template(..., tokenize=True) puede, según
    la versión de transformers/tokenizers instalada, devolver un objeto crudo
    tokenizers.Encoding en vez de una lista plana de ints en ciertos casos límite
    (rompe silenciosamente recién al escribir a Arrow, no al tokenizar). Para
    evitar esa ambigüedad, usamos SIEMPRE tokenize=False (que devuelve un string,
    estable en todas las versiones) y tokenizamos ese string por separado."""
    messages = ejemplo["messages"]

    def ids_de(msgs):
        if not msgs:
            return []
        texto = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return tok(texto, add_special_tokens=False)["input_ids"]

    full_ids = ids_de(messages)
    labels = [IGNORE_INDEX] * len(full_ids)

    prev_len = 0
    for i in range(1, len(messages) + 1):
        curr_len = len(ids_de(messages[:i]))
        rol = messages[i - 1]["role"]
        if rol == "assistant":
            # Los tokens nuevos que aparecieron al agregar este mensaje SÍ cuentan
            # para el loss (son los que el modelo debe aprender a generar).
            for j in range(prev_len, min(curr_len, len(labels))):
                labels[j] = full_ids[j]
        prev_len = curr_len

    full_ids = full_ids[:max_len]
    labels = labels[:max_len]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class DataCollatorConversacional:
    """Padding dinámico por batch. input_ids se rellenan con pad_token_id,
    labels se rellenan con -100 (para que el padding no afecte el loss)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = self.tok.pad_token_id or self.tok.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            n_pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n_pad)
            attention_mask.append(b["attention_mask"] + [0] * n_pad)
            labels.append(b["labels"] + [IGNORE_INDEX] * n_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main(args):
    print(f"🔧 Cargando tokenizer y modelo base '{args.base_model}' en 4-bit...")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    modelo = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto",
    )
    modelo = prepare_model_for_kbit_training(modelo)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    modelo = get_peft_model(modelo, lora_config)
    modelo.print_trainable_parameters()

    print(f"📂 Cargando datos: {args.train} / {args.val}")
    train_raw = cargar_jsonl(args.train)
    val_raw = cargar_jsonl(args.val)
    print(f"   Train: {len(train_raw)} | Val: {len(val_raw)}")

    print("🔤 Tokenizando con máscara de loss solo en turnos 'assistant'...")
    train_ds = Dataset.from_list(train_raw).map(
        lambda ej: tokenizar_con_mascara(ej, tok, args.max_len), remove_columns=["messages", "esi_level"])
    val_ds = Dataset.from_list(val_raw).map(
        lambda ej: tokenizar_con_mascara(ej, tok, args.max_len), remove_columns=["messages", "esi_level"])

    collator = DataCollatorConversacional(tok)

    training_args = TrainingArguments(
        output_dir=args.checkpoints_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("\n🚀 Entrenando...")
    trainer.train()

    print(f"\n💾 Guardando modelo final en '{args.out}'...")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    print(f"\n✅ Listo. Modelo fine-tuneado guardado en: {args.out}")
    print("   Próximo paso: apuntar server_chat_v2.py a esta carpeta y correrlo.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data_v2/train.jsonl")
    p.add_argument("--val", default="data_v2/val.jsonl")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct",
                    help="Debe ser el MISMO modelo (o de la misma familia/chat template) "
                         "usado en 2_generar_dataset_llm.py, para que el formato de chat coincida")
    p.add_argument("--out", default="trained_model/final_model")
    p.add_argument("--checkpoints-dir", default="checkpoints_qlora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=1024)
    main(p.parse_args())