"""
server_chat_gguf.py - Misma API que server_chat_v2.py, pero sirviendo el
modelo cuantizado en GGUF vía llama-cpp-python en vez de transformers/peft.

Por qué existe este archivo aparte (y no se pisa server_chat_v2.py):
  - La lógica de la conversación (system prompt, parseo de "Triage: N - ...",
    MAX_TURNOS, concurrencia con semáforo + threadpool) es IDÉNTICA a la
    versión con transformers — lo único que cambia es CÓMO se carga y se
    llama al modelo.
  - Requiere un archivo .gguf ya fusionado y cuantizado (ver
    convert_to_gguf.sh) — no lee una carpeta de adapter LoRA ni un modelo
    base de Hugging Face.

Uso:
  python server_chat_gguf.py --gguf qwen7b-triage-Q4_K_M.gguf
  http://localhost:8000
"""

import argparse
import asyncio
import os
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from llama_cpp import Llama
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", default="model.gguf",
                    help="Ruta al archivo .gguf ya fusionado y cuantizado "
                         "(ver convert_to_gguf.sh).")
    p.add_argument("--n-ctx", type=int, default=4096,
                    help="Tamaño de contexto (tokens). 4096 alcanza de sobra "
                         "para el system prompt + varios turnos de esta app.")
    p.add_argument("--threads", type=int, default=os.cpu_count(),
                    help="Threads de CPU para inferencia. Default: todos los "
                         "disponibles en la máquina/contenedor.")
    # Railway/Heroku inyectan el puerto real vía $PORT — si existe, manda sobre
    # el default. Local: sigue arrancando en 8000 si no seteás nada.
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    return p.parse_args()


_args = parse_args()

# ====== CONFIGURACIÓN ======
GGUF_PATH = Path(_args.gguf)
STATIC_DIR = Path("./static_chat")
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ESI Triage Chat API (GGUF)", version="2.0")

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

# Solo una llamada de inferencia a la vez sobre esta instancia del modelo —
# misma razón que en server_chat_v2.py: liberar el event loop sin arriesgar
# llamadas concurrentes sobre el mismo objeto Llama.
semaforo_generacion = asyncio.Semaphore(1)


# ====== MODELOS Pydantic ======
class ChatMessage(BaseModel):
    """Mensaje entrante del chat."""
    user_message: str
    conversation_state: dict  # {"messages": [...]}  (vacío/None en el primer turno)


# ====== CARGAR MODELO ======
print(f"[INFO] Cargando modelo GGUF desde '{GGUF_PATH}' ({_args.threads} threads)...")
try:
    llm = Llama(
        model_path=str(GGUF_PATH),
        n_ctx=_args.n_ctx,
        n_threads=_args.threads,
        verbose=False,
        # Sin chat_format explícito: llama-cpp-python toma el chat template
        # embebido en el propio .gguf (el mismo que usaba el tokenizer del
        # modelo fusionado) — es el que realmente vio el modelo en el ajuste
        # fino, así que es más seguro que forzar un chat_format genérico acá.
    )
    print("[INFO] ✅ Modelo cargado\n")
except Exception as e:
    print(f"[ERROR] No se pudo cargar modelo: {e}")
    llm = None


# ====== GENERACIÓN CONVERSACIONAL ======
def generar_respuesta_modelo(messages: list, max_tokens: int = 200) -> str:
    """Le pasa el historial completo de mensajes al modelo (mismo formato que se usó
    para entrenar) y devuelve el turno siguiente: o una pregunta, o la clasificación
    final ('Triage: N - ...'). Función SÍNCRONA y bloqueante a propósito — se llama
    desde un threadpool (ver /chat), no directamente desde una corrutina."""
    if llm is None:
        raise RuntimeError("Modelo no cargado")

    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,   # determinístico en producción, igual que la versión transformers
        top_p=0.9,
    )
    return out["choices"][0]["message"]["content"].strip()


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
    return {"status": "ok", "model_loaded": llm is not None}


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
        # esté realmente usando el modelo — con varios usuarios, las
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
    print("🏥 SERVIDOR CHAT (GGUF) - ESI TRIAGE (conversación real turno a turno)")
    print("=" * 70)
    print(f"\n📍 http://localhost:{_args.port}  (modelo: {GGUF_PATH})")
    print("💬 El modelo decide cuántas preguntas hacer (0/1/2) y cuándo clasificar\n")

    uvicorn.run(app, host="0.0.0.0", port=_args.port)