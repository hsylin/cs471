#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=01:00:00
#SBATCH --job-name=skewbench
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository directory}"
source scripts/cluster-env.sh
mkdir -p "$SKB_ROOT/results/hardware"
nvidia-smi -q > "$SKB_ROOT/results/hardware/nvidia-smi-${SLURM_JOB_ID}.txt"
nvidia-smi topo -m > "$SKB_ROOT/results/hardware/topology-${SLURM_JOB_ID}.txt"
python -c 'import torch,vllm; print({"torch":torch.__version__,"cuda":torch.version.cuda,"vllm":vllm.__version__,"gpus":torch.cuda.device_count()}); assert torch.cuda.device_count()==2'
exec torchrun --standalone --nproc-per-node=2 -m skewbench.runner "$@"
