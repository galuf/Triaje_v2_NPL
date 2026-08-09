# Imagen slim: acá NO hace falta CUDA porque Railway/Heroku no dan GPU.
FROM python:3.11-slim

WORKDIR /app

# Instalar Git LFS para descargar archivos grandes correctamente
RUN apt-get update && apt-get install -y git-lfs && apt-get clean && rm -rf /var/lib/apt/lists/*

# torch CPU-only — evita bajar la build con CUDA (mucho más pesada e inútil acá).
# --default-timeout y --retries: la descarga son ~190MB; con conexión lenta o
# inestable, el timeout corto por default de pip corta la descarga a mitad de
# camino. Le damos más margen para que no falle por eso.
RUN pip install --default-timeout=1000 --retries 10 --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements_cpu.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY server_chat_v2.py .
COPY static_chat/ ./static_chat/
COPY trained_model_qwen_1.5b/ ./trained_model_qwen_1.5b/

# Precarga el modelo base DURANTE el build, no en el arranque del contenedor.
# Sin esto, cada vez que el dyno/servicio reinicia tendría que bajar ~3GB
# desde Hugging Face Hub — en Heroku eso puede superar el timeout de boot
# (60s) y tirar un error R10. Horneándolo en la imagen, el arranque solo
# carga desde disco local, mucho más rápido y confiable.
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')"

# Exponer puertos: 8080 (Cloud Run default), 8000 (local/Railway)
EXPOSE 8000 8080

# Cloud Run, Railway, Heroku inyectan el puerto vía $PORT
# El servidor lo lee automáticamente desde la variable de entorno
CMD ["python", "server_chat_v2.py", "--adapter", "trained_model_qwen_1.5b"]