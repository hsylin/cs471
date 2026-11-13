#!/bin/bash
#SBATCH --account=cs471
#SBATCH --qos=cs471
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=fullmoe-eight-48h
#SBATCH --output=/work/cs-471/skewbench-hlin/logs/fullmoe-eight-%j.log
set -euo pipefail
state="/work/cs-471/skewbench-hlin/reservations/fullmoe-eight-${SLURM_JOB_ID}"
mkdir -p "$state"
scontrol show job "$SLURM_JOB_ID" > "$state/allocation.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$state/nodes.txt"
date --iso-8601=seconds > "$state/started.txt"
srun --input=none --ntasks=2 --ntasks-per-node=1 bash -c 'hostname; nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv; nvidia-smi topo -m' > "$state/hardware.txt"
sleep 172800
