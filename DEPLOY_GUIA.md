# 📦 Guía de Despliegue con Git LFS

## ✅ Paso 1: Agregar archivos a Git (YA HECHO)
Git LFS ya está configurado y trackeando `*.safetensors` y `*.json`

## 📤 Paso 2: Agregar todo a Git

```bash
cd /home/galuf/Escritorio/triagechatbotfinal_v3

# Agregar archivos (Git LFS automáticamente manejará los grandes)
git add .
git add -A

# Configurar usuario (si no lo has hecho)
git config user.email "tu_email@gmail.com"
git config user.name "Tu Nombre"

# Hacer commit
git commit -m "Initial commit with LFS models"
```

## 🌐 Paso 3: Crear Repositorio en GitHub

1. Ve a https://github.com/new
2. Crea un repositorio llamado `triagechatbot`
3. NO inicialices con README
4. Copia el comando que te muestra (algo como):

```bash
git remote add origin https://github.com/TU_USUARIO/triagechatbot.git
git branch -M main
git push -u origin main
```

## 🚀 Paso 4: Deploy en Railway

1. Ve a https://railway.app
2. Haz login con GitHub
3. Crea nuevo proyecto → Deploy desde GitHub
4. Selecciona `triagechatbot`
5. Railway detecta el Dockerfile automáticamente
6. ✅ ¡Deploy completo!

## ⚙️ Configuración en Railway (IMPORTANTE)

En la dashboard de Railway:
- Ir a Variables de Entorno
- Agregar si es necesario: `PORT=8000`

## 🔍 Solución de Problemas

### Si falla el deploy con error de Git LFS:
```bash
# En tu máquina local:
git lfs pull --all
git push --force origin main
```

### Si Railway no descarga los archivos LFS:
- Railway soporta LFS, pero asegúrate de que Git LFS esté instalado ✅ (ya lo hicimos en Dockerfile)

## 📋 Resumen de lo que hicimos:

✅ Instalamos Git LFS  
✅ Trackeamos archivos grandes (.safetensors, .json)  
✅ Creamos .gitignore para no subir caché innecesario  
✅ Actualizamos Dockerfile para instalar Git LFS en el contenedor  

## 🎯 Siguiente: Subir a GitHub

Espera confirmación y luego ejecuta los comandos del Paso 2 y 3.
