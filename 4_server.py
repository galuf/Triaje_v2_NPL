"""
4_server.py
Backend FastAPI para el chatbot (compatible con tu index.html).
Endpoint: POST /chat  →  { "response": str, "derivacion": {...} }

Flujo por mensaje:
  1. RAG (TF-IDF) encuentra el patrón más parecido → diagnóstico probable + especialidad.
  2. El modelo fine-tuneado predice el triage con el prompt en formato de entrenamiento.
  3. Se arma la tarjeta de derivación que renderiza el frontend.

Si aún NO entrenaste el modelo, corre igual en modo solo-RAG (fallback),
así puedes probar el frontend desde ya.

Uso:
  pip install fastapi uvicorn
  python 4_server.py --adapter modelo_triage        # con modelo
  python 4_server.py --sin-modelo                   # solo RAG
"""
import argparse
import json
import pickle
import re

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from config_triage import (ESI_LABELS, ESPECIALIDAD, DIAGNOSTICO, INSTRUCCION,
                           SYSTEM_PROMPT, respuesta_referencia)

# ── Estado global ─────────────────────────────────────────────────────────────
RAG = {}
MODELO = {}

app = FastAPI(title="Triage Virtual API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    messages: list  # [{role, content}, ...]


def cargar_rag(rag_dir):
    RAG["casos"]  = json.load(open(f"{rag_dir}/rag_ed_casos.json", encoding="utf-8"))
    RAG["vec"]    = pickle.load(open(f"{rag_dir}/rag_ed_vectorizer.pkl", "rb"))
    RAG["matrix"] = np.load(f"{rag_dir}/rag_ed_matrix.npy")
    print(f"✅ RAG cargado: {len(RAG['casos'])} patrones")


def cargar_modelo(adapter, base_model):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    else:
        print("⚠ Sin GPU: el modelo correrá en CPU (lento pero funciona con 1B)")

    MODELO["tok"] = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    MODELO["model"] = PeftModel.from_pretrained(base, adapter)
    MODELO["model"].eval()
    print(f"✅ Modelo fine-tuneado cargado desde '{adapter}'")


def buscar_rag(texto):
    """Devuelve (caso_top, similitud)."""
    q = RAG["vec"].transform([texto.lower()])
    sims = cosine_similarity(q, RAG["matrix"])[0]
    idx = int(np.argmax(sims))
    return RAG["casos"][idx], float(sims[idx])


def generar_con_modelo(texto_usuario):
    import torch
    tok, model = MODELO["tok"], MODELO["model"]
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Motivo de consulta: {texto_usuario}. "
                                      f"Signos vitales: no disponibles."},
    ]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=70, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def parse_nivel(texto):
    m = re.search(r"[Tt]riage:?\s*([1-5])", texto)
    return int(m.group(1)) if m else None
def falta_duracion_intensidad(texto):
    """True si el texto no menciona ni duración ni intensidad del síntoma."""
    duracion = re.search(
        r'\b(hace|desde|llevo)\b.{0,20}\b(minuto|hora|d[ií]a|semana|mes)',
        texto, re.IGNORECASE)
    intensidad = re.search(
        r'\b(leve|moderad[oa]|fuerte|intens[oa]|insoportable)\b|'
        r'\b([1-9]|10)\s*(de\s*10|/\s*10|\s*puntos)',
        texto, re.IGNORECASE)
    return not (duracion or intensidad)

@app.post("/chat")
def chat(req: ChatRequest):
    user_msgs = [m["content"] for m in req.messages if m["role"] == "user"]
    texto = " . ".join(user_msgs)

    # 1) RAG: patrón más parecido
    caso, sim = buscar_rag(texto)
    complaint = caso["chief_complaint_en"]

    # Si la similitud es muy baja, pedir más información en vez de adivinar
    if sim < 0.05:
        return {
            "response": "Entiendo. ¿Podrías contarme un poco más? Por ejemplo: "
                        "¿dónde te duele, desde cuándo, y qué tan fuerte del 1 al 10?",
            "derivacion": None,
        }
        # 1.5) Una sola pregunta de seguimiento si falta duración/intensidad  ← NUEVO
    if len(user_msgs) == 1 and falta_duracion_intensidad(texto):
        return {
            "response": "Para orientarte mejor: ¿desde cuándo tienes esto, "
                        "y qué tan fuerte es del 1 al 10?",
            "derivacion": None,
        }
    # 2) Predicción de triage
    if MODELO:
        salida = generar_con_modelo(texto)
        nivel = parse_nivel(salida) or caso["esi_level"]  # fallback al RAG si no parsea
    else:
        salida = respuesta_referencia(complaint, caso["esi_level"])
        nivel = caso["esi_level"]

    # 3) Tarjeta de derivación (formato que espera index.html)
    derivacion = {
        "derivacion": True,
        "especialidad": ESPECIALIDAD.get(complaint, "MED.EMER. Y DESASTRES"),
        "nivel": nivel,
        "triage": ESI_LABELS.get(nivel, "Urgente"),
        "diagnostico_probable": DIAGNOSTICO.get(complaint, "Por evaluar"),
        "instruccion": INSTRUCCION.get(nivel, ""),
    }

    respuesta = (f"Gracias por contarme. Según lo que describes:\n\n{salida}\n\n"
                 f"{INSTRUCCION.get(nivel, '')}")
    if nivel <= 2:
        respuesta += "\n\n🚨 Tu caso es prioritario: avisa de inmediato al personal de la clínica."

    return {"response": respuesta, "derivacion": derivacion}


@app.get("/")
def root():
    return {"status": "ok", "modelo_cargado": bool(MODELO), "rag_cargado": bool(RAG)}


if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--rag-dir", default=".")
    p.add_argument("--adapter", default="modelo_triage")
    p.add_argument("--base-model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--sin-modelo", action="store_true", help="Modo solo RAG (sin fine-tuning)")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    cargar_rag(a.rag_dir)
    if not a.sin_modelo:
        try:
            cargar_modelo(a.adapter, a.base_model)
        except Exception as e:
            print(f"⚠ No se pudo cargar el modelo ({e}). Arrancando en modo solo-RAG.")

    uvicorn.run(app, host="0.0.0.0", port=a.port)
