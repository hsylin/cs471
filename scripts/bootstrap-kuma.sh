#!/usr/bin/env bash
set -euo pipefail
export SKB_ROOT="${SKB_ROOT:-/work/cs-471/skewbench-hlin}"
umask 077
mkdir -p "$SKB_ROOT"/{cache,logs,models,results}
export UV_CACHE_DIR="$SKB_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$SKB_ROOT/python"
uv venv --python 3.12.14 "$SKB_ROOT/venv"
uv pip install --python "$SKB_ROOT/venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match \
  'https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  numpy matplotlib pytest huggingface-hub safetensors
# Pin the matching CUDA build: PyPI's newer codec can require CUDA 13 even
# when vLLM and torch were installed from the CUDA 12.9 distribution.
uv pip install --python "$SKB_ROOT/venv/bin/python" \
  --index-url https://download.pytorch.org/whl/cu129 'torchcodec==0.16.0+cu129'
uv pip freeze --python "$SKB_ROOT/venv/bin/python" > "$SKB_ROOT/results/environment-lock.txt"

