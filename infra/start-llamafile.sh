#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"
export MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models}"
export PORT="${PORT:-8080}"
export LLAMAFILE_VERSION="${LLAMAFILE_VERSION:-0.10.0}"

exec "$SCRIPT_DIR/run-llamafile.sh"
