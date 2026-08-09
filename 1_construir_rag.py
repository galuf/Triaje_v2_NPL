"""
1_construir_rag.py
Construye el índice RAG (chunks + embeddings) a partir del manual ESI en PDF.
Este índice lo consume 2_generar_dataset_llm.py para fundamentar la
justificación clínica de cada ejemplo generado. En producción (server_chat_v2.py)
NO se usa — el modelo fine-tuneado ya aprendió ese razonamiento.

Uso:
  python 1_construir_rag.py --pdf ESI.pdf --out esi_index

Salida (en --out):
  chunks.json              lista de fragmentos de texto
  embeddings.npy           embeddings de cada fragmento (misma cantidad y orden)
  modelo_embeddings.txt    nombre del modelo de sentence-transformers usado
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_PALABRAS = 180     # palabras por fragmento
OVERLAP_PALABRAS = 40    # solapamiento entre fragmentos consecutivos
MIN_PALABRAS_CHUNK = 40  # descarta fragmentos demasiado cortos (ruido de portada, etc.)


def limpiar_texto(texto: str) -> str:
    """pypdf extrae el PDF de ESI con un salto de línea por casi cada palabra
    (o incluso por sílaba, por cómo está tipografiado el PDF original). Hay
    que colapsar todo a texto corrido antes de poder trocearlo con sentido."""
    texto = texto.replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def es_chunk_referencias(texto: str) -> bool:
    """Descarta fragmentos que son mayormente listas bibliográficas (DOIs,
    años entre paréntesis en cadena) — no aportan al razonamiento clínico
    y solo consumen espacio del índice RAG."""
    dois = texto.lower().count("doi.org")
    return dois >= 2


def trocear(texto: str, chunk_palabras: int, overlap: int) -> list:
    palabras = texto.split(" ")
    chunks = []
    i = 0
    while i < len(palabras):
        trozo = palabras[i:i + chunk_palabras]
        if len(trozo) >= MIN_PALABRAS_CHUNK:
            chunk_texto = " ".join(trozo)
            if not es_chunk_referencias(chunk_texto):
                chunks.append(chunk_texto)
        i += chunk_palabras - overlap
    return chunks


def main(args):
    print(f"📂 Leyendo PDF: {args.pdf}")
    reader = PdfReader(args.pdf)
    print(f"   {len(reader.pages)} páginas")

    texto_completo = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        limpio = limpiar_texto(raw)
        if limpio:
            texto_completo.append(limpio)

    texto_unido = " ".join(texto_completo)
    print(f"   {len(texto_unido.split())} palabras totales tras limpieza")

    print(f"✂️  Troceando en fragmentos de ~{CHUNK_PALABRAS} palabras "
          f"(solapamiento {OVERLAP_PALABRAS})...")
    chunks = trocear(texto_unido, CHUNK_PALABRAS, OVERLAP_PALABRAS)
    print(f"   {len(chunks)} fragmentos útiles (se descartaron los que eran "
          f"mayormente referencias bibliográficas)")

    print(f"🔧 Cargando modelo de embeddings '{MODELO_EMBEDDINGS}'...")
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)

    print("🧮 Calculando embeddings...")
    embeddings = modelo.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    np.save(out / "embeddings.npy", embeddings)
    (out / "modelo_embeddings.txt").write_text(MODELO_EMBEDDINGS)

    print(f"\n✅ Índice RAG listo en '{out}/':")
    print(f"   - chunks.json       ({len(chunks)} fragmentos)")
    print(f"   - embeddings.npy    (shape {embeddings.shape})")
    print(f"   - modelo_embeddings.txt")
    print(f"\nPreview del primer fragmento:\n   {chunks[0][:200]}...")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", default="ESI.pdf", help="Ruta al manual ESI en PDF")
    p.add_argument("--out", default="esi_index", help="Carpeta de salida del índice")
    main(p.parse_args())
    