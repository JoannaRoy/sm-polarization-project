#!/usr/bin/env bash
set -euo pipefail

: "${INSTALL_DIR:?INSTALL_DIR must be set}"
: "${MODEL_DIR:?MODEL_DIR must be set}"
: "${PORT:?PORT must be set}"

LLAMAFILE_VERSION="${LLAMAFILE_VERSION:-0.10.0}"
MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODEL_FILE="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLAMAFILE_URL="https://github.com/mozilla-ai/llamafile/releases/download/${LLAMAFILE_VERSION}/llamafile-${LLAMAFILE_VERSION}"
LLAMAFILE_BIN="${INSTALL_DIR}/llamafile-${LLAMAFILE_VERSION}"

mkdir -p "$MODEL_DIR" "$INSTALL_DIR"

if [ ! -f "$LLAMAFILE_BIN" ]; then
  echo "Downloading llamafile ${LLAMAFILE_VERSION}..."
  curl -fsSL -o "$LLAMAFILE_BIN" "$LLAMAFILE_URL"
  chmod +x "$LLAMAFILE_BIN"
  echo "llamafile downloaded."
else
  echo "llamafile binary already exists, skipping download."
fi

if [ ! -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
  echo "Downloading model (this may take a while)..."
  if [ -n "${HF_TOKEN:-}" ]; then
    curl -fsSL -H "Authorization: Bearer ${HF_TOKEN}" -o "${MODEL_DIR}/${MODEL_FILE}" "$MODEL_URL"
  else
    curl -fsSL -o "${MODEL_DIR}/${MODEL_FILE}" "$MODEL_URL"
  fi
  echo "Model downloaded."
else
  echo "Model already exists, skipping download."
fi

echo "Starting llamafile on port ${PORT}..."
exec "$LLAMAFILE_BIN" \
  --server \
  -m "${MODEL_DIR}/${MODEL_FILE}" \
  --host 0.0.0.0 \
  --port "$PORT" \
  -c 4096
