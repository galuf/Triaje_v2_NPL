"""
2_generar_dataset_llm.py
Genera data/train.jsonl, val.jsonl, test.jsonl usando un LLM (<3B) + RAG sobre el
handbook ESI para fundamentar el lenguaje clínico.

IMPORTANTE sobre la arquitectura:
  - El LLM + RAG se usan SOLO ACÁ, para generar datos de entrenamiento (offline).
  - El nivel ESI y los signos vitales de cada ejemplo son REALES (vienen del CSV).
  - El LLM genera: la frase del paciente, las preguntas de la enfermera y la
    justificación clínica (fundamentada en fragmentos del handbook vía RAG).
  - Las RESPUESTAS del paciente (qué signos vitales reporta) son SIEMPRE
    determinísticas — no las genera el LLM — para evitar que invente datos que
    el paciente no tiene o conteste algo que no coincide con la pregunta.
  - El modelo final (2_train_qlora.py) NO usa RAG — aprende este comportamiento
    vía fine-tuning. En producción (4_server.py) no se consulta el handbook.
  - Casos ESI 1 ("bandera roja"): se clasifican directo, sin preguntar — igual
    que haría una enfermera ante una emergencia obvia.
  - Cantidad de preguntas variable (0/1/2): en la vida real el paciente a veces
    da toda la información de una, a veces hace falta insistir — no siempre es
    "una pregunta, una respuesta, listo".

Uso:
  # Prueba piloto rápida (pocos ejemplos, para revisar calidad)
  python 2_generar_dataset_llm.py --por-clase 4 --out data_piloto

  # Corrida moderada
  python 2_generar_dataset_llm.py --por-clase 150 --out data_v2
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from config_triage import (MOTIVO_ES, VARIANTES, JUSTIFICACION, ESPECIALIDAD,
                            ESI_LABELS, hallazgos_vitales)

RANDOM_SEED = 42

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

# Probabilidad de que un paciente en su casa tenga cada instrumento — la mayoría NO tiene
# oxímetro ni tensiómetro, y nadie mide su propia frecuencia respiratoria de forma confiable.
PROB_TERMOMETRO = 0.65
PROB_OXIMETRO = 0.15
PROB_TENSIOMETRO = 0.20
PROB_RELOJ_FC = 0.30  # smartwatch/pulsera con medición de pulso

# Cuántas veces pregunta la enfermera antes de clasificar (además de bandera roja, que es 0
# siempre) — en la vida real no siempre alcanza con una pregunta, ni siempre hace falta.
PROB_CERO_PREGUNTAS = 0.15
PROB_DOS_PREGUNTAS = 0.30


def simular_instrumentos(rng):
    return {
        "termometro": rng.random() < PROB_TERMOMETRO,
        "oximetro": rng.random() < PROB_OXIMETRO,
        "tensiometro": rng.random() < PROB_TENSIOMETRO,
        "reloj_fc": rng.random() < PROB_RELOJ_FC,
    }


def decidir_num_rondas(rng, es_bandera_roja):
    """0 = el motivo ya alcanza para clasificar; 1 = una pregunta y listo (el caso más
    común); 2 = la primera respuesta no fue suficiente, hace falta insistir."""
    if es_bandera_roja:
        return 0
    r = rng.random()
    if r < PROB_CERO_PREGUNTAS:
        return 0
    if r < PROB_CERO_PREGUNTAS + PROB_DOS_PREGUNTAS:
        return 2
    return 1


def _disponible(row, campo):
    """True solo si el valor existe y NO es NaN (pandas usa NaN para valores faltantes,
    no None — 'is not None' no lo detecta y deja pasar 'nan' literal al texto)."""
    return pd.notna(row.get(campo))


def _partes_vitales(row, instrumentos):
    """Separa lo que el paciente podría reportar en dos grupos: básicos (dolor, pulso —
    lo primero que se pregunta) e instrumentos médicos (termómetro/oxímetro/tensiómetro —
    lo que se pregunta si la primera respuesta no alcanzó)."""
    basicos = []
    if _disponible(row, "pain_score"):
        basicos.append(f"Dolor {row['pain_score']:.0f}/10")
    if instrumentos["reloj_fc"] and _disponible(row, "heart_rate"):
        basicos.append(f"FC {row['heart_rate']:.0f} (según reloj)")

    medicos = []
    if instrumentos["termometro"] and _disponible(row, "temperature"):
        medicos.append(f"Temp {row['temperature']:.1f}°C")
    if instrumentos["oximetro"] and _disponible(row, "spo2"):
        medicos.append(f"SpO2 {row['spo2']:.0f}%")
    if instrumentos["tensiometro"] and _disponible(row, "systolic_bp") and _disponible(row, "diastolic_bp"):
        medicos.append(f"PA {row['systolic_bp']:.0f}/{row['diastolic_bp']:.0f}")

    faltantes_pulso = [] if instrumentos["reloj_fc"] else ["forma de medir el pulso exacto"]
    faltantes_medicos = []
    if not instrumentos["oximetro"]:
        faltantes_medicos.append("oxímetro")
    if not instrumentos["tensiometro"]:
        faltantes_medicos.append("tensiómetro")

    return basicos, medicos, faltantes_pulso, faltantes_medicos


def _armar_texto(partes, faltantes):
    if not partes and not faltantes:
        return "No tengo nada más para agregar."
    texto = f"Signos vitales: {', '.join(partes)}." if partes else ""
    if faltantes:
        texto += f" No tengo {' ni '.join(faltantes)}."
    return texto.strip()


def construir_texto_vitales_disponibles(row, instrumentos):
    """Todo en una sola respuesta (para 1 ronda de preguntas)."""
    basicos, medicos, falt_pulso, falt_medicos = _partes_vitales(row, instrumentos)
    return _armar_texto(basicos + medicos, falt_pulso + falt_medicos)


def construir_texto_vitales_dos_rondas(row, instrumentos):
    """Divide la info en dos respuestas — a veces el paciente no dice todo de una.
    Ronda 1: dolor y pulso (si tiene reloj). Ronda 2: termómetro/oxímetro/tensiómetro."""
    basicos, medicos, falt_pulso, falt_medicos = _partes_vitales(row, instrumentos)
    texto1 = _armar_texto(basicos, falt_pulso)
    texto2 = _armar_texto(medicos, falt_medicos)
    return texto1, texto2


def construir_prompt(motivo_en, motivo_es, edad, vitales_texto, hallazgos, esi_level,
                      esi_label, es_bandera_roja, num_rondas, fragmentos):
    if num_rondas == 0:
        instrucciones_preguntas = ('(Este caso no necesita preguntas de seguimiento — el '
                                     'motivo ya alcanza para clasificar con criterio clínico razonable.)')
    elif num_rondas == 1:
        instrucciones_preguntas = ('- "pregunta_enfermera": UNA pregunta breve que haría la '
                                     'enfermera para obtener signos vitales o intensidad del dolor.')
    else:
        instrucciones_preguntas = (
            '- "pregunta_enfermera": una PRIMERA pregunta breve sobre dolor o signos vitales.\n'
            '- "pregunta_enfermera_2": una SEGUNDA pregunta de seguimiento — a veces el paciente '
            'no da toda la información de una — preguntá específicamente por algo que no haya '
            'quedado claro (ej. si tiene oxímetro o termómetro en casa).'
        )

    vitales_linea = ""
    if es_bandera_roja:
        vitales_linea = f"\n- Signos vitales reales (caso crítico, SÍ podés citarlos): {vitales_texto}"

    hallazgos_texto = ", ".join(hallazgos) if hallazgos else "ninguno relevante"

    return f"""Sos un asistente que ayuda a construir datos de entrenamiento para un chatbot \
de enfermería de triage de emergencias, en español. Te doy información real de un paciente y \
fragmentos del manual oficial ESI (Emergency Severity Index). Generá SOLO un JSON con estas \
claves, sin explicación adicional:

- "frase_paciente": cómo diría el paciente su motivo de consulta, en español coloquial, corto.
{instrucciones_preguntas}
- "justificacion": UNA oración breve en español justificando el nivel de triage, basada en los \
fragmentos del manual ESI que te paso. IMPORTANTE: NO inventes números exactos de signos \
vitales que no te di abajo — usá hallazgos cualitativos (ej. "taquicardia", sin decir "FC \
136") salvo que te haya dado "signos vitales reales". NO inventes partes del cuerpo, órganos, \
mecanismos de lesión (ej. "trauma torácico", "trauma pulmonar") ni diagnósticos que no estén \
relacionados con el motivo de consulta real — los fragmentos son solo para el RAZONAMIENTO \
clínico (por qué ese nivel de urgencia), no para agregar síntomas o lesiones que el paciente \
no tiene. NO menciones "los fragmentos", "el manual", "el RAG" ni el proceso que usaste para \
razonar — escribí la justificación como si fuera tu propio criterio clínico de enfermera, no \
una cita de una fuente. Si el cuadro no es claramente leve y no vas a tener SpO2 confirmado, \
mencioná que se recomienda confirmar en persona.

Datos del caso:
- Motivo (inglés): {motivo_en}
- Motivo (español): {motivo_es}
- Edad: {edad}
- Hallazgos anormales (podés mencionar estos SIN número exacto): {hallazgos_texto}
- Nivel ESI real: {esi_level} ({esi_label}){vitales_linea}

Fragmentos del manual ESI (en inglés, para fundamentar la justificación clínica):
{fragmentos}

Respondé SOLO con el JSON, nada más."""


def cargar_indice_rag(idx_dir):
    idx_dir = Path(idx_dir)
    chunks = json.load(open(idx_dir / "chunks.json", encoding="utf-8"))
    embeddings = np.load(idx_dir / "embeddings.npy")
    modelo_nombre = open(idx_dir / "modelo_embeddings.txt").read().strip()
    modelo_emb = SentenceTransformer(modelo_nombre)
    return chunks, embeddings, modelo_emb


def buscar_fragmentos(query, chunks, embeddings, modelo_emb, top_k=3):
    q_emb = modelo_emb.encode([query], convert_to_numpy=True)
    sims = (embeddings @ q_emb.T).flatten() / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in idx]


def extraer_json(texto):
    """Busca el primer bloque {...} balanceado en el texto generado."""
    start = texto.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(texto[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(texto[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


PALABRAS_NORMALIDAD = ["normal", "normales", "dentro de parámetros", "sin alteraciones"]
PALABRAS_PREOCUPACION = ["preocupante", "preocupantes", "anormal", "anormales", "alto riesgo",
                          "alta seguridad", "vigilancia estrecha", "elevado", "elevados", "grave"]
PALABRAS_CONTRADICCION_CRITICO = ["no necesita hospital", "no necesariamente implica",
                                    "no es necesario ingresar", "puede esperar",
                                    "no es urgente", "no urgente", "no requiere atención inmediata"]
# Mismas etiquetas que devuelve hallazgos_vitales() — si el LLM las menciona en su
# propio texto Y al mismo tiempo dice "normal", se contradice a sí mismo sin importar
# lo que diga la data real (ej. "presenta taquicardia... sin signos vitales fuera de
# los límites normales" en la misma oración).
ETIQUETAS_HALLAZGOS = ["taquicardia", "bradicardia", "desaturación", "fiebre alta",
                        "fiebre", "taquipnea", "hipotensión", "dolor intenso"]
PALABRAS_METAREFERENCIA = ["los fragmentos", "el fragmento", "según el manual esi",
                             "según los fragmentos", "el manual esi indica",
                             "de acuerdo con el manual", "el rag", "la búsqueda"]

# Negaciones comunes en español clínico — si una palabra de alarma aparece justo
# después de una de estas, NO cuenta como afirmación (ej. "sin hallazgos anormales"
# significa que NO hay nada anormal, no que sí lo haya). Técnica clásica de NLP
# clínico para detectar hallazgos negados (cf. algoritmo NegEx).
NEGACIONES = ["sin ", "no ", "ausencia de ", "no hay ", "ningún ", "ninguna ",
              "no presenta", "no muestra", "no necesariamente", "no requiere",
              "no es ", "no son ", "descarta", "descartar", "la falta de", "falta de"]
_VENTANA_NEGACION = 50


def _mencion_afirmada(texto_lower, palabra):
    """True si `palabra` aparece en el texto SIN estar negada justo antes, y
    respetando límites de palabra (para que 'normales' no matchee dentro de
    'anormales', que la contiene como substring)."""
    for m in re.finditer(r"(?<![a-záéíóúñ])" + re.escape(palabra) + r"(?![a-záéíóúñ])", texto_lower):
        contexto_previo = texto_lower[max(0, m.start() - _VENTANA_NEGACION):m.start()]
        if not any(neg in contexto_previo for neg in NEGACIONES):
            return True
    return False


def justificacion_menciona_nivel_distinto(justificacion, esi_level):
    """El RAG a veces trae fragmentos del handbook que hablan de OTRO nivel ESI (ej.
    un fragmento sobre criterios de ESI 2 para un caso que en realidad es ESI 3), y el
    LLM termina citando ese número dentro de su propia justificación — una
    contradicción dentro del mismo mensaje ('Triage: 3 ... sugiere nivel ESI 2...').
    Detecta cualquier mención de 'ESI <n>' que no coincida con el nivel real."""
    menciones = re.findall(r"esi\D{0,15}([1-5])\b", justificacion.lower())
    return any(int(n) != esi_level for n in menciones)


def justificacion_inconsistencia(justificacion, hallazgos, esi_level, es_bandera_roja=False):
    """Devuelve una etiqueta corta del tipo de contradicción detectada (para poder
    diagnosticar qué validación está rechazando más ejemplos), o None si no hay
    ninguna: decir 'normal' cuando hay hallazgos anormales reales, decir 'preocupante'
    en un caso que la data real dice que NO es urgente (ESI 4-5) sin que
    hallazgos_vitales() respalde esa alarma, citar un número de nivel ESI distinto al
    real, o — en bandera roja (ESI 1) — decir que no hace falta ir al hospital.

    Ojo: en ESI 1-3 SÍ es normal que la justificación suene grave/de alto riesgo sin
    que hallazgos_vitales() haya detectado nada — el nivel ESI real puede ser alto por
    el motivo o mecanismo de consulta (ej. dificultad respiratoria, hemorragia) aunque
    los números de esta fila no crucen los umbrales fijos de esa función. Por eso este
    chequeo solo aplica a ESI 4-5, donde SÍ sería una contradicción real."""
    texto = justificacion.lower()
    dice_normal = any(_mencion_afirmada(texto, p) for p in PALABRAS_NORMALIDAD)
    dice_preocupante = any(_mencion_afirmada(texto, p) for p in PALABRAS_PREOCUPACION)
    menciona_hallazgo_propio = any(_mencion_afirmada(texto, e) for e in ETIQUETAS_HALLAZGOS)
    if dice_normal and (hallazgos or menciona_hallazgo_propio):
        return "dice_normal_con_hallazgos"
    if dice_preocupante and not hallazgos and esi_level >= 4:
        return "dice_preocupante_sin_hallazgos"
    if justificacion_menciona_nivel_distinto(justificacion, esi_level):
        return "nivel_esi_contradictorio"
    if es_bandera_roja and any(p in texto for p in PALABRAS_CONTRADICCION_CRITICO):
        return "contradice_bandera_roja"
    if any(p in texto for p in PALABRAS_METAREFERENCIA):
        return "menciona_fragmentos_del_prompt"
    return None


def generar_con_llm(tok, modelo, prompt, max_new_tokens=300):
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(modelo.device)
    with torch.no_grad():
        out = modelo.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                              temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def construir_fallback(motivo_en, motivo_es, hallazgos, esi_level, num_rondas):
    """Si el LLM falla en generar JSON válido, usamos las plantillas fijas como respaldo."""
    just = JUSTIFICACION.get(motivo_en, "Cuadro que requiere evaluación")
    extra = f", con {' y '.join(hallazgos)}" if hallazgos else ""
    datos = {
        "frase_paciente": motivo_es,
        "justificacion": f"{just}{extra}",
    }
    if num_rondas >= 1:
        datos["pregunta_enfermera"] = ("¿Podrías darme algunos datos más: intensidad del "
                                         "dolor y si tenés algún signo vital disponible?")
    if num_rondas == 2:
        datos["pregunta_enfermera_2"] = ("¿Tenés termómetro, oxímetro o tensiómetro para "
                                           "confirmar algún otro dato?")
    return datos


def construir_ejemplo(row, tok, modelo, chunks, embeddings, modelo_emb, rng, fallos):
    complaint_en = row["chief_complaint"]
    complaint_es = MOTIVO_ES.get(complaint_en, complaint_en.lower())
    esi_level = int(row["esi_level"])
    es_bandera_roja = esi_level == 1

    hallazgos = hallazgos_vitales(
        hr=row.get("heart_rate"), rr=row.get("respiratory_rate"),
        temp=row.get("temperature"), spo2=row.get("spo2"),
        sbp=row.get("systolic_bp"), pain=row.get("pain_score"),
    )

    partes_reales = []
    if _disponible(row, "heart_rate"): partes_reales.append(f"FC {row['heart_rate']:.0f}")
    if _disponible(row, "systolic_bp") and _disponible(row, "diastolic_bp"):
        partes_reales.append(f"PA {row['systolic_bp']:.0f}/{row['diastolic_bp']:.0f}")
    if _disponible(row, "respiratory_rate"): partes_reales.append(f"FR {row['respiratory_rate']:.0f}")
    if _disponible(row, "temperature"): partes_reales.append(f"Temp {row['temperature']:.1f}°C")
    if _disponible(row, "spo2"): partes_reales.append(f"SpO2 {row['spo2']:.0f}%")
    if _disponible(row, "pain_score"): partes_reales.append(f"Dolor {row['pain_score']:.0f}/10")
    vitales_texto = ", ".join(partes_reales) if partes_reales else "no registrados"

    instrumentos = simular_instrumentos(rng)
    num_rondas = decidir_num_rondas(rng, es_bandera_roja)

    respuesta_1 = respuesta_2 = ""
    if num_rondas == 1:
        respuesta_1 = construir_texto_vitales_disponibles(row, instrumentos)
    elif num_rondas == 2:
        respuesta_1, respuesta_2 = construir_texto_vitales_dos_rondas(row, instrumentos)

    fragmentos = buscar_fragmentos(complaint_en, chunks, embeddings, modelo_emb, top_k=3)

    prompt = construir_prompt(complaint_en, complaint_es, row.get("age", "N/A"), vitales_texto,
                                hallazgos, esi_level, ESI_LABELS[esi_level], es_bandera_roja,
                                num_rondas, "\n---\n".join(fragmentos))

    salida = generar_con_llm(tok, modelo, prompt)
    datos = extraer_json(salida)

    razones = []
    if datos is None:
        razones.append("json_invalido")
    else:
        if not datos.get("frase_paciente"):
            razones.append("frase_paciente_vacia")
        if not datos.get("justificacion"):
            razones.append("justificacion_vacia")
        if num_rondas >= 1 and not datos.get("pregunta_enfermera"):
            razones.append("falta_pregunta_1")
        if num_rondas == 2 and not datos.get("pregunta_enfermera_2"):
            razones.append("falta_pregunta_2")
        if datos.get("justificacion"):
            motivo_inconsistencia = justificacion_inconsistencia(
                datos["justificacion"], hallazgos, esi_level, es_bandera_roja)
            if motivo_inconsistencia:
                razones.append(motivo_inconsistencia)
            if (not es_bandera_roja and not instrumentos["oximetro"]
                    and re.search(r"spo2\D{0,5}\d", datos["justificacion"].lower())):
                razones.append("spo2_filtrado")

    if razones:
        fallos["total"] += 1
        for r in razones:
            fallos[r] = fallos.get(r, 0) + 1
        rechazados = fallos.setdefault("_rechazados", [])
        rechazados.append({
            "motivo": complaint_en, "esi_level": esi_level, "razones": razones,
            "justificacion_original": datos.get("justificacion") if datos else None,
        })
        datos = construir_fallback(complaint_en, complaint_es, hallazgos, esi_level, num_rondas)

    especialidad = ESPECIALIDAD.get(complaint_en, "MED.EMER. Y DESASTRES")
    # El LLM a veces ya termina la justificación en "." (o en "‥"/"…", que Qwen genera
    # a veces como caracteres Unicode de puntuación en vez de ".") y algunas
    # especialidades del diccionario ya traen "." en la abreviatura (ej. "MED. FAM. Y
    # COMUNIT.") — sin limpiar esto quedaba ".. Derivar a ..." con punto doble.
    CARACTERES_FINALES = " .‥…"  # espacio, punto, two-dot-leader (‥), elipsis (…)
    justificacion_limpia = datos["justificacion"].strip().rstrip(CARACTERES_FINALES)
    especialidad_limpia = especialidad.strip().rstrip(CARACTERES_FINALES)
    clasificacion_final = (f"Triage: {esi_level} - {ESI_LABELS[esi_level]}. "
                            f"{justificacion_limpia}. Derivar a {especialidad_limpia}.")

    # Red de seguridad determinística: si falta el oxímetro (el signo vital más crítico del
    # algoritmo ESI, junto a FC/FR) y el cuadro no es claramente leve, recomendar evaluación
    # presencial. Solo aplica si de verdad se simuló una conversación sobre instrumentos.
    if num_rondas >= 1 and not instrumentos["oximetro"] and esi_level <= 3:
        clasificacion_final += (" No se pudo confirmar la saturación de oxígeno en casa — "
                                  "se recomienda acudir cuanto antes para medir los signos vitales.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT_V2},
                {"role": "user", "content": f"Motivo de consulta: {datos['frase_paciente']}."}]

    if num_rondas == 0:
        messages.append({"role": "assistant", "content": clasificacion_final})
    elif num_rondas == 1:
        messages.append({"role": "assistant", "content": datos["pregunta_enfermera"]})
        messages.append({"role": "user", "content": respuesta_1})
        messages.append({"role": "assistant", "content": clasificacion_final})
    else:  # 2 rondas
        messages.append({"role": "assistant", "content": datos["pregunta_enfermera"]})
        messages.append({"role": "user", "content": respuesta_1})
        messages.append({"role": "assistant", "content": datos["pregunta_enfermera_2"]})
        messages.append({"role": "user", "content": respuesta_2})
        messages.append({"role": "assistant", "content": clasificacion_final})

    return {"messages": messages, "esi_level": esi_level}


def main(args):
    rng = random.Random(RANDOM_SEED)

    print(f"🔧 Cargando índice RAG desde '{args.rag_dir}'...")
    chunks, embeddings, modelo_emb = cargar_indice_rag(args.rag_dir)
    print(f"   {len(chunks)} fragmentos disponibles")

    print(f"🔧 Cargando LLM generador '{args.model}'...")
    tok = AutoTokenizer.from_pretrained(args.model)
    modelo = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    modelo.eval()

    print(f"📂 Cargando {args.csv}...")
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["clinical_notes", "esi_level", "chief_complaint"]).copy()
    df["esi_level"] = df["esi_level"].astype(int)

    partes = []
    for _, grupo in df.groupby("esi_level"):
        n = min(args.por_clase, len(grupo))
        partes.append(grupo.sample(n=n, random_state=RANDOM_SEED))
    df_muestra = pd.concat(partes).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"   {len(df_muestra)} filas a generar ({args.por_clase} por clase, máx.)")

    fallos = {"total": 0}
    ejemplos = []
    for i, row in df_muestra.iterrows():
        ej = construir_ejemplo(row, tok, modelo, chunks, embeddings, modelo_emb, rng, fallos)
        ejemplos.append(ej)
        if (i + 1) % 10 == 0 or (i + 1) == len(df_muestra):
            print(f"   Generados {i + 1}/{len(df_muestra)} (fallos LLM→fallback: {fallos['total']})")

    df_ej = pd.DataFrame({"esi_level": [e["esi_level"] for e in ejemplos]})

    def split_seguro(indices, test_size, y):
        """Split estratificado si hay suficientes datos por clase; si no (ej. piloto
        chico), cae a un split simple sin estratificar para no romper con pocos datos."""
        try:
            return train_test_split(indices, test_size=test_size, stratify=y, random_state=RANDOM_SEED)
        except ValueError:
            print(f"   ⚠️  Muy pocos ejemplos para estratificar por clase — split simple sin estratificar.")
            return train_test_split(indices, test_size=test_size, random_state=RANDOM_SEED)

    idx_train, idx_temp = split_seguro(df_ej.index, 0.2, df_ej["esi_level"])
    idx_val, idx_test = split_seguro(
        df_ej.loc[idx_temp].index, 0.5, df_ej.loc[idx_temp, "esi_level"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for nombre, idxs in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        with open(out / f"{nombre}.jsonl", "w", encoding="utf-8") as f:
            for i in idxs:
                f.write(json.dumps(ejemplos[i], ensure_ascii=False) + "\n")

    print(f"\n✅ Listo. Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")
    print(f"   Fallback a plantilla fija: {fallos['total']}/{len(ejemplos)} "
          f"({100*fallos['total']/len(ejemplos):.1f}%)")
    razones_detalle = {k: v for k, v in fallos.items() if k not in ("total", "_rechazados")}
    if razones_detalle:
        print("   Motivos de fallback (un ejemplo puede tener más de uno):")
        for razon, cuenta in sorted(razones_detalle.items(), key=lambda kv: -kv[1]):
            print(f"      - {razon}: {cuenta}")

    rechazados = fallos.get("_rechazados", [])
    if rechazados:
        print(f"\n🔍 Texto ORIGINAL del LLM que fue rechazado (para revisar si el filtro es "
              f"justo o demasiado estricto):")
        for r in rechazados:
            print(f"   [ESI {r['esi_level']}, {r['motivo']}] razones={r['razones']}")
            print(f"      -> {r['justificacion_original']}")

    print(f"\n📋 Preview de 2 ejemplos generados:")
    for ej in ejemplos[:2]:
        print(json.dumps(ej, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="fedmml_ed_triage_dataset.csv")
    p.add_argument("--rag-dir", default="esi_index")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                    help="LLM usado SOLO para generar datos (no necesariamente el que se fine-tunea)")
    p.add_argument("--por-clase", type=int, default=20,
                    help="Ejemplos a generar por nivel ESI")
    p.add_argument("--out", default="data_v2")
    main(p.parse_args())
