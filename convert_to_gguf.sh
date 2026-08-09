# Instalar prerrequisitos (una vez):
pip install torch transformers peft --break-system-packages
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt --break-system-packages
cmake -B build && cmake --build build --config Release -j
cd ..

# Correr la conversión:
chmod +x convert_to_gguf.sh
./convert_to_gguf.sh trained_model_qwen7b/final_model Qwen/Qwen2.5-7B-Instruct qwen7b-triage