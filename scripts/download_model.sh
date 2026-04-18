#!/bin/bash
# Download LLaMA-3.2-3B-Instruct GGUF (Q4_K_M) — fits in 4GB VRAM
set -e

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

MODEL_FILE="$MODELS_DIR/llama-3.2-3b-instruct-q4_k_m.gguf"
if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists: $MODEL_FILE"
    exit 0
fi

echo "Downloading LLaMA-3.2-3B-Instruct Q4_K_M (~2.0 GB)..."
# Hugging Face direct link
wget -c \
  "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  -O "$MODEL_FILE"

echo "Model saved to: $MODEL_FILE"
echo "Set NOUS_MODEL_PATH=$MODEL_FILE or it will be found automatically."
