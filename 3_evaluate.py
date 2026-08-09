"""
3_evaluate.py
Evalúa sobre data/test.jsonl y compara sistemas:
  --sistema modelo → modelo fine-tuneado (QLoRA)
  --sistema rag    → baseline RAG TF-IDF (requiere archivos de build_rag.py)

Métricas: accuracy, F1 macro, matriz de confusión, tasa de sub-triage, BLEU.

Uso:
  python 3_evaluate.py --sistema rag
  python 3_evaluate.py --sistema modelo --adapter modelo_triage
"""
import argparse
import json
import re

import numpy as np

from config_triage import ESI_LABELS, respuesta_referencia, SYSTEM_PROMPT_V2


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def parse_nivel(texto):
    """Extrae el nivel ESI de 'Triage: N - ...'. Devuelve None si no se puede."""
    m = re.search(r"[Tt]riage:?\s*([1-5])", texto)
    return int(m.group(1)) if m else None


def mensaje_clasificacion(messages):
    """El último mensaje del asistente es siempre la clasificación final — con el
    formato nuevo (data_v2) la conversación puede tener 3, 5 o 7 mensajes según
    cuántas rondas de preguntas hizo la enfermera, así que no está en un índice fijo."""
    for m in reversed(messages):
        if m["role"] == "assistant":
            return m["content"]
    return ""


# ── Sistema 1: modelo fine-tuneado ────────────────────────────────────────────
def predecir_modelo(test, adapter, base_model, max_ejemplos, max_rondas=2):
    """Simula la conversación completa turno a turno: si el modelo pregunta en vez
    de clasificar, le respondemos con el turno de usuario REAL del ejemplo (los
    signos vitales determinísticos ya generados), igual que hablaría el paciente —
    así el modelo nunca tiene que inventar esa respuesta, solo decidir si preguntar
    y cuándo clasificar. Devuelve también el tiempo por encuentro (s), para poder
    reportarlo con el mismo formato que la Tabla 1 del paper de referencia."""
    import time
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from tqdm import tqdm

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb,
                                                 device_map="auto")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    generados = []
    tiempos = []
    for ej in tqdm(test[:max_ejemplos], desc="Generando"):
        inicio = time.time()
        turnos_usuario_reales = [m["content"] for m in ej["messages"] if m["role"] == "user"]
        conversacion = [{"role": "system", "content": SYSTEM_PROMPT_V2},
                        {"role": "user", "content": turnos_usuario_reales[0]}]
        siguiente_turno = 1
        texto = ""

        for _ in range(max_rondas + 1):  # como mucho max_rondas preguntas + la clasificación
            inputs = tokenizer.apply_chat_template(
                conversacion,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            texto = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            if parse_nivel(texto) is not None or siguiente_turno >= len(turnos_usuario_reales):
                break  # clasificó, o ya no quedan turnos reales del paciente para responderle

            conversacion.append({"role": "assistant", "content": texto})
            conversacion.append({"role": "user", "content": turnos_usuario_reales[siguiente_turno]})
            siguiente_turno += 1

        generados.append(texto)
        tiempos.append(time.time() - inicio)

    return generados, tiempos

# ── Sistema 2: baseline RAG (TF-IDF) ─────────────────────────────────────────
def predecir_rag(test, rag_dir, max_ejemplos):
    import time
    import pickle
    from sklearn.metrics.pairwise import cosine_similarity

    casos  = json.load(open(f"{rag_dir}/rag_ed_casos.json", encoding="utf-8"))
    vec    = pickle.load(open(f"{rag_dir}/rag_ed_vectorizer.pkl", "rb"))
    matrix = np.load(f"{rag_dir}/rag_ed_matrix.npy")

    generados = []
    tiempos = []
    for ej in test[:max_ejemplos]:
        inicio = time.time()
        consulta = ej["messages"][1]["content"].lower()
        q = vec.transform([consulta])
        sims = cosine_similarity(q, matrix)[0]
        top = casos[int(np.argmax(sims))]
        # El baseline responde con la misma plantilla → BLEU comparable
        generados.append(respuesta_referencia(top["chief_complaint_en"], top["esi_level"]))
        tiempos.append(time.time() - inicio)
    return generados, tiempos

# ── Sistema 3: modelo nativo (entrenado desde cero) ──────────────────────────
def predecir_nativo(test, native_dir, max_ejemplos, max_rondas=2):
    import time
    from pathlib import Path
    import torch
    from tokenizers import Tokenizer
    from modelo_nativo import GPTNativo

    native_path = Path(native_dir)
    tok = Tokenizer.from_file(str(native_path / "tokenizer.json"))
    checkpoint = torch.load(native_path / "modelo.pt", map_location="cpu")

    # Reconstruir la arquitectura desde la configuración guardada
    cfg = checkpoint["config"]
    modelo = GPTNativo(
        vocab_size=checkpoint["vocab_size"],
        n_capas=cfg["n_capas"],
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        n_cabezas=cfg["n_cabezas"],
        max_len=checkpoint["max_len"]
    )
    modelo.load_state_dict(checkpoint["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(device)
    modelo.eval()

    # Mapeo de tokens especiales
    bos_id = tok.token_to_id("<|bos|>")
    sys_id = tok.token_to_id("<|system|>")
    user_id = tok.token_to_id("<|user|>")
    asst_id = tok.token_to_id("<|assistant|>")
    end_id = tok.token_to_id("<|endofturn|>")

    generados = []
    tiempos = []

    for ej in test[:max_ejemplos]:
        inicio = time.time()
        turnos_usuario = [m["content"] for m in ej["messages"] if m["role"] == "user"]
        
        # Iniciar la secuencia de tokens con el System Prompt
        ids = [bos_id, sys_id] + tok.encode(SYSTEM_PROMPT_V2).ids + [end_id]
        ids += [user_id] + tok.encode(turnos_usuario[0]).ids + [end_id]
        ids += [asst_id]

        siguiente_turno = 1
        texto = ""

        for _ in range(max_rondas + 1):
            input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
            len_prompt = input_tensor.shape[1]

            with torch.no_grad():
                out_ids = modelo.generar(
                    input_tensor,
                    max_new_tokens=150,
                    eos_id=end_id,
                    temperatura=0.0
                )[0]

            # Decodificar solo los tokens nuevos generados
            nuevos_ids = out_ids[len_prompt:].tolist()
            texto = tok.decode(nuevos_ids).strip()

            # Si ya clasificó o no quedan más respuestas reales del paciente, salir
            if parse_nivel(texto) is not None or siguiente_turno >= len(turnos_usuario):
                break

            # Agregar el turno del asistente y la siguiente respuesta del usuario
            ids = out_ids.tolist()
            ids += [user_id] + tok.encode(turnos_usuario[siguiente_turno]).ids + [end_id] + [asst_id]
            siguiente_turno += 1

        generados.append(texto)
        tiempos.append(time.time() - inicio)

    return generados, tiempos
# ── Métricas propias (las que ya veníamos usando) ─────────────────────────────
def evaluar_nuestras_metricas(test, generados):
    from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                                 classification_report)
    import sacrebleu

    n = len(generados)
    y_true = [ej["esi_level"] for ej in test[:n]]
    y_pred = [parse_nivel(g) for g in generados]

    validos = [i for i, p in enumerate(y_pred) if p is not None]
    tasa_parseo = len(validos) / n
    yt = [y_true[i] for i in validos]
    yp = [y_pred[i] for i in validos]

    print("=" * 60)
    print("NUESTRAS MÉTRICAS (accuracy/F1 solo sobre salidas parseables)")
    print("=" * 60)
    print(f"Ejemplos evaluados: {n}  |  Salidas parseables: {tasa_parseo:.1%}")


    # 🛑 CONTROL DE SEGURIDAD: Evita el ValueError si no hay predicciones parseables
    if len(validos) == 0:
        print("⚠️  No hay ninguna salida parseable con el formato 'Triage: N'. Métricas omitidas.")
        return
    print(f"Accuracy : {accuracy_score(yt, yp):.4f}")
    print(f"F1 macro : {f1_score(yt, yp, average='macro'):.4f}")

    # Sub-triage: predecir MENOS urgente que lo real (número mayor) → error grave
    sub = sum(1 for t, p in zip(yt, yp) if p > t) / len(yt)
    sub_grave = sum(1 for t, p in zip(yt, yp) if t <= 2 and p > t) / max(1, sum(1 for t in yt if t <= 2))
    print(f"Sub-triage total          : {sub:.1%}")
    print(f"Sub-triage en ESI 1-2 (⚠) : {sub_grave:.1%}")

    print("\nMatriz de confusión (filas=real, columnas=predicho, niveles 1-5):")
    print(confusion_matrix(yt, yp, labels=[1, 2, 3, 4, 5]))
    print("\n" + classification_report(yt, yp, labels=[1, 2, 3, 4, 5],
          target_names=[f"{k} {v}" for k, v in ESI_LABELS.items()], zero_division=0))

    # BLEU/ROUGE-L: pedidas explícitamente por el profesor — miden similitud de
    # redacción del texto generado contra la referencia, no la corrección clínica.
    referencias = [mensaje_clasificacion(ej["messages"]) for ej in test[:n]]
    bleu = sacrebleu.corpus_bleu(generados, [referencias])
    print(f"BLEU   : {bleu.score:.2f}")
    try:
        from rouge_score import rouge_scorer
        rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rl = np.mean([rs.score(r, g)["rougeL"].fmeasure for r, g in zip(referencias, generados)])
        print(f"ROUGE-L: {rl:.4f}")
    except ImportError:
        pass


# ── Métricas al estilo del paper de referencia ────────────────────────────────
# (Aljohani et al., "Domain-Adapted Small Language Models for Reliable Clinical
# Triage", arXiv:2604.26766) — mismas fórmulas exactas (ecuaciones 1-5 del paper),
# calculadas sobre el TOTAL de ejemplos (una salida no-parseable cuenta como
# discordancia, igual que cualquier otra falla de clasificación en la práctica).
def evaluar_estilo_paper(test, generados, tiempos):
    n = len(generados)
    y_true = [ej["esi_level"] for ej in test[:n]]
    y_pred = [parse_nivel(g) for g in generados]

    # Ec. 1: Total Discordance = Misclasificaciones / Total (no-parseable = error)
    misclasificados = sum(1 for t, p in zip(y_true, y_pred) if p is None or p != t)
    discordancia = misclasificados / n

    # Ec. 2-3: Under/Overtriage = predicho > / < real, sobre el TOTAL de textos
    undertriage = sum(1 for t, p in zip(y_true, y_pred) if p is not None and p > t) / n
    overtriage  = sum(1 for t, p in zip(y_true, y_pred) if p is not None and p < t) / n

    # Ec. 4: Significant Undertriage = real=2, predicho en {3,4,5}
    sig_under = sum(1 for t, p in zip(y_true, y_pred)
                     if t == 2 and p in (3, 4, 5)) / n
    # Ec. 5: Significant Overtriage = real en {3,4,5}, predicho en {1,2}
    sig_over = sum(1 for t, p in zip(y_true, y_pred)
                    if t in (3, 4, 5) and p in (1, 2)) / n

    tiempo_prom = sum(tiempos) / len(tiempos) if tiempos else float("nan")

    print("=" * 60)
    print("MÉTRICAS ESTILO PAPER (Aljohani et al. 2026) — sobre el TOTAL de ejemplos")
    print("=" * 60)
    print(f"Total Discordance   : {discordancia:.2%}")
    print(f"Undertriage         : {undertriage:.2%}")
    print(f"Overtriage          : {overtriage:.2%}")
    print(f"Significant Under   : {sig_under:.2%}")
    print(f"Significant Over    : {sig_over:.2%}")
    print(f"Time (s/encuentro)  : {tiempo_prom:.2f}")
    print("\nPara comparar contra la Tabla 1/3 del paper (mejor Qwen2.5-7B fine-tuneado):")
    print("  Discordance=25.85% | Under=13.35% | Over=12.50% | Sig.Under=6.25% | "
          "Sig.Over=2.56% | Time=0.16s")





def main(args):
    test = load_jsonl(f"{args.data}/test.jsonl")
    
    if args.sistema == "modelo":
        generados, tiempos = predecir_modelo(test, args.adapter, args.base_model, args.max_ejemplos)
    elif args.sistema == "nativo":
        generados, tiempos = predecir_nativo(test, args.native_dir, args.max_ejemplos)
    else:
        generados, tiempos = predecir_rag(test, args.rag_dir, args.max_ejemplos)

    print(f"\n🔎 Sistema evaluado: {args.sistema.upper()}")
    for g in generados[:3]:
        print("  →", g)
    print()
    evaluar_nuestras_metricas(test, generados)
    print()
    evaluar_estilo_paper(test, generados, tiempos)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sistema", choices=["modelo", "rag", "nativo"], required=True) # <-- Agregado 'nativo'
    p.add_argument("--data", default="data_v2")
    p.add_argument("--adapter", default="modelo_v2_esi")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--native-dir", default="trained_native/base") # <-- Nuevo argumento
    p.add_argument("--rag-dir", default=".")
    p.add_argument("--max-ejemplos", type=int, default=500)
    main(p.parse_args())