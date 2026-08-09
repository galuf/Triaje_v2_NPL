"""
server_chat_v2.py - API FastAPI con conversación REAL turno a turno

Diferencia clave respecto a la v1:
  - Ya NO hay un formulario fijo de 10 preguntas en orden.
  - El modelo fine-tuneado decide, en cada turno, si:
        a) clasifica directo (respuesta empieza con "Triage: N - ...")
        b) hace una pregunta más (0, 1 o 2 rondas, según aprendió en el fine-tuning)
  - Esto es exactamente el comportamiento que se generó en 2_generar_dataset_llm.py:
    ahí NO había un formulario, había una lista de mensajes system/user/assistant.
    Este servidor reproduce ese mismo formato en producción.
  - En producción NO se usa RAG (igual que el diseño original): el modelo ya
    aprendió a razonar clínicamente vía fine-tuning, sin consultar el handbook.

Concurrencia:
  - generar_respuesta_modelo() es una llamada bloqueante (torch.generate). Se
    ejecuta en un threadpool (run_in_threadpool) para no congelar el event loop
    de FastAPI mientras genera — así el proceso sigue respondiendo a otros
    endpoints (ej. /health) mientras un usuario espera su turno.
  - Un asyncio.Semaphore(1) asegura que nunca se llame a generate() dos veces
    en simultáneo sobre la misma instancia del modelo (una sola GPU, un solo
    modelo cargado — no hay paralelismo real posible ni deseable acá). Con
    múltiples usuarios, las peticiones se atienden una por una, en orden, sin
    que el servidor se cuelgue por completo mientras tanto.

Uso:
  python server_chat_v2.py
  http://localhost:8000
"""

import argparse
import asyncio
import os
import re
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from peft import AutoPeftModelForCausalLM
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="trained_model/final_model",
                    help="Carpeta del adapter LoRA (debe tener adapter_config.json). "
                         "NO hace falta pasar --base-model: AutoPeftModelForCausalLM "
                         "lo lee solo desde adapter_config.json.")
    # Railway/Heroku/Google Cloud Run inyectan el puerto real vía $PORT
    # Default 8080 para Cloud Run compatibility, 8000 para local/Railway
    default_port = int(os.environ.get("PORT", 8080))
    p.add_argument("--port", type=int, default=default_port)
    return p.parse_args()


_args = parse_args()

# ====== CONFIGURACIÓN ======
MODEL_DIR = Path(_args.adapter)
STATIC_DIR = Path("./static_chat")
STATIC_DIR.mkdir(exist_ok=True)

# float16 solo tiene sentido (y buen soporte de kernels) en GPU. En CPU
# (Railway/Heroku, sin GPU) hay que usar float32 — float16 en CPU suele ser
# más lento o directamente no soportado para ciertas operaciones.
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

app = FastAPI(title="ESI Triage Chat API", version="2.0")

# Debe ser IDÉNTICO al usado en 2_generar_dataset_llm.py — el modelo fue
# fine-tuneado esperando este texto exacto como mensaje "system".
SYSTEM_PROMPT_V2 = (
    "Eres una enfermera de triage de emergencias. Si el cuadro presenta signos de alarma "
    "evidentes (paro cardíaco, no responde, convulsión activa, hemorragia masiva), "
    "clasificá de inmediato sin preguntar. En los demás casos, hacé las preguntas breves "
    "que necesites para conocer los signos vitales o la intensidad del dolor si no te los "
    "dieron — a veces alcanza con una, a veces el paciente no da todo de una y hace falta "
    "insistir. El paciente está en su casa, antes de llegar a la clínica — puede no tener "
    "oxímetro, tensiómetro ni forma de medir su frecuencia cardíaca o respiratoria con "
    "precisión. Si falta un signo vital importante (sobre todo el oxímetro) y el cuadro no "
    "es claramente leve, recomendá que acuda a que se lo midan en persona en vez de asumir "
    "que está bien. Cuando tengas suficiente información, respondé SIEMPRE con este formato "
    "exacto: 'Triage: <nivel 1-5> - <etiqueta>. <justificación breve>. Derivar a <especialidad>.'"
)

MAX_TURNOS = 5  # tope de seguridad: si el modelo nunca clasifica, cortamos igual

# Solo un generate() a la vez sobre esta instancia del modelo — ver nota de
# concurrencia arriba.
semaforo_generacion = asyncio.Semaphore(1)


# ====== MODELOS Pydantic ======
class ChatMessage(BaseModel):
    """Mensaje entrante del chat."""
    user_message: str
    conversation_state: dict  # {"messages": [...]}  (vacío/None en el primer turno)


# ====== CARGAR MODELO ======
print("[INFO] Cargando modelo fine-tuneado...")
try:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.eval()
    print("[INFO] ✅ Modelo cargado\n")
except Exception as e:
    print(f"[ERROR] No se pudo cargar modelo: {e}")
    model = None
    tokenizer = None


# ====== GENERACIÓN CONVERSACIONAL ======
def generar_respuesta_modelo(messages: list, max_new_tokens: int = 200) -> str:
    """Le pasa el historial completo de mensajes al modelo (mismo formato que se usó
    para entrenar) y devuelve el turno siguiente: o una pregunta, o la clasificación
    final ('Triage: N - ...'). Función SÍNCRONA y bloqueante a propósito — se llama
    desde un threadpool (ver /chat), no directamente desde una corrutina."""
    if model is None or tokenizer is None:
        raise RuntimeError("Modelo no cargado")

    # return_dict=True fuerza siempre un BatchEncoding (input_ids + attention_mask),
    # sin importar la versión de transformers instalada — evita el bug de intentar
    # llamar .shape sobre un objeto que a veces no es un tensor plano.
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # determinístico en producción (a diferencia de la
            temperature=0.1,       # generación del dataset, que usaba sampling para variedad)
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ====== PARSEO DE LA CLASIFICACIÓN FINAL ======
TRIAGE_RE = re.compile(
    r"triage:\s*([1-5])\s*-\s*(.+?)\.\s*(.+?)\.\s*derivar a\s*(.+?)\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def parsear_triage(texto: str) -> dict | None:
    """Si el texto generado es la clasificación final, extrae sus partes.
    Si no matchea, quiere decir que el modelo todavía está preguntando algo."""
    m = TRIAGE_RE.search(texto)
    if not m:
        return None
    return {
        "esi": int(m.group(1)),
        "etiqueta": m.group(2).strip(),
        "justificacion": m.group(3).strip(),
        "especialidad": m.group(4).strip(),
    }


def get_esi_info(esi: int) -> dict:
    info = {
        1: {"name": "🚨 Resucitación Inmediata",
            "description": "Situación de emergencia crítica. Requiere intervención inmediata.",
            "color": "#ff4444", "action": "LLAMAR AL 117 AHORA - Está en peligro inmediato"},
        2: {"name": "🔴 Emergencia de Alto Riesgo",
            "description": "Paciente con síntomas graves que pueden deteriorarse rápidamente.",
            "color": "#ff8c00", "action": "IR INMEDIATAMENTE A EMERGENCIA - Avisa al personal"},
        3: {"name": "🟡 Urgencia - Moderada",
            "description": "Requiere evaluación urgente. Probablemente necesites varios exámenes.",
            "color": "#ffdd00", "action": "DIRIGIRSE A EMERGENCIA - Se atenderá pronto"},
        4: {"name": "🟢 Urgencia Menor",
            "description": "Síntoma que requiere atención pero no es crítico.",
            "color": "#90ee90", "action": "ESPERAR EN EMERGENCIA - Será atendido"},
        5: {"name": "🟢 No Urgente",
            "description": "Síntomas leves que pueden tratarse ambulatoriamente.",
            "color": "#00aa00", "action": "CONSULTA EXTERNA - No es emergencia"},
    }
    return info.get(esi, info[3])


# ====== ENDPOINTS ======
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("./static_chat/chat.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/chat")
async def chat(msg: ChatMessage):
    """Un solo endpoint conversacional: no hay 'steps' fijos.
    El front-end solo manda el mensaje del usuario + el historial que le devolvimos
    la vez anterior (conversation_state.messages)."""

    messages = msg.conversation_state.get("messages")

    if not messages:
        # Primer turno: arrancamos igual que en 2_generar_dataset_llm.py
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V2},
            {"role": "user", "content": f"Motivo de consulta: {msg.user_message.strip()}."},
        ]
    else:
        messages.append({"role": "user", "content": msg.user_message.strip()})

    # Tope de seguridad: si ya hubo demasiadas rondas y el modelo no clasificó,
    # forzamos a que el próximo turno sea la clasificación (no debería pasar casi
    # nunca si el fine-tuning salió bien, pero evita loops infinitos).
    turnos_assistant = sum(1 for m in messages if m["role"] == "assistant")
    if turnos_assistant >= MAX_TURNOS:
        messages.append({
            "role": "user",
            "content": "(No tengo más información. Por favor clasificá con lo que ya tenés.)",
        })

    try:
        # run_in_threadpool libera el event loop mientras se genera (no congela
        # el servidor entero). El semáforo asegura que solo un thread a la vez
        # esté realmente usando la GPU/el modelo — con varios usuarios, las
        # peticiones se atienden una por una, en orden.
        async with semaforo_generacion:
            respuesta = await run_in_threadpool(generar_respuesta_modelo, messages)
    except Exception as e:
        return {"error": f"Error al generar respuesta: {str(e)}"}

    messages.append({"role": "assistant", "content": respuesta})

    triage = parsear_triage(respuesta)

    if triage:
        esi_info = get_esi_info(triage["esi"])
        return {
            "done": True,
            "response": respuesta,
            "esi": triage["esi"],
            "esi_info": esi_info,
            "state": {"messages": messages},
        }

    # El modelo todavía está preguntando algo (0, 1 o 2 rondas — lo decide él solo)
    return {
        "done": False,
        "response": respuesta,
        "state": {"messages": messages},
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("🏥 SERVIDOR CHAT v2 - ESI TRIAGE (conversación real turno a turno)")
    print("=" * 70)
    print(f"\n📍 http://localhost:{_args.port}  (adapter: {MODEL_DIR})")
    print("💬 El modelo decide cuántas preguntas hacer (0/1/2) y cuándo clasificar\n")

    uvicorn.run(app, host="0.0.0.0", port=_args.port)