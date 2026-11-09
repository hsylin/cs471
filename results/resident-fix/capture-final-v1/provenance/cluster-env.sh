#!/usr/bin/env bash
# Source from a Slurm job on Kuma; credentials are never stored here.
export SKB_ROOT="${SKB_ROOT:-/work/cs-471/skewbench-hlin}"
export UV_CACHE_DIR="$SKB_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$SKB_ROOT/python"
export HF_HOME="$SKB_ROOT/cache/huggingface"
export TRITON_CACHE_DIR="$SKB_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$SKB_ROOT/cache/vllm"
export XDG_CACHE_HOME="$SKB_ROOT/cache/xdg"
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
source "$SKB_ROOT/venv/bin/activate"

