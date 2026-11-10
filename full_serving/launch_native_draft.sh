#!/usr/bin/env bash
# Unexecuted full-model launcher draft. Default: print only; never allocates GPUs.
# Run once per node inside the existing allocation, with all four GPUs visible.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: launch_native_draft.sh off|on primary|worker MODEL_SNAPSHOT PRIMARY_IP [--execute]

Default is a dry run. MODEL_SNAPSHOT must be the complete pinned local snapshot.
Start the primary and worker commands in separate Slurm steps on their respective
nodes. This script does not submit jobs, choose nodes, open tunnels or download data.
The initial backend is allgather_reducescatter, not optimized DeepEP all-to-all.
EOF
}

if (( $# < 4 || $# > 5 )); then usage >&2; exit 2; fi
arm=$1
role=$2
model_snapshot=$3
primary_ip=$4
execute=${5:-}
[[ $arm == off || $arm == on ]] || { usage >&2; exit 2; }
[[ $role == primary || $role == worker ]] || { usage >&2; exit 2; }
[[ -z $execute || $execute == --execute ]] || { usage >&2; exit 2; }

command=(vllm serve "$model_snapshot"
  --served-model-name qwen3-full
  --dtype bfloat16
  --max-model-len 32768
  --generation-config vllm
  --tensor-parallel-size 1
  --data-parallel-size 8
  --data-parallel-size-local 4
  --enable-expert-parallel
  --all2all-backend allgather_reducescatter
  --data-parallel-address "$primary_ip"
  --data-parallel-rpc-port 13345)

if [[ $arm == on ]]; then
  command+=(--enable-eplb --eplb-config '{"num_redundant_experts":32}')
fi
if [[ $role == primary ]]; then
  command+=(--host "$primary_ip" --port 8000 --api-server-count 4)
else
  command+=(--data-parallel-start-rank 4 --headless)
fi

printf 'Draft command (%s, %s):\n' "$arm" "$role"
printf '%q ' "${command[@]}"
printf '\n'
[[ $execute == --execute ]] || exit 0

[[ ${SLURM_JOB_ID:-} == 4882284 ]] || {
  printf 'Refusing launch outside reservation 4882284.\n' >&2; exit 1;
}
[[ ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}} == 2 ]] || {
  printf 'Expected a two-node allocation.\n' >&2; exit 1;
}
[[ ${model_snapshot##*/} == ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 ]] || {
  printf 'Expected the pinned checkpoint snapshot directory.\n' >&2; exit 1;
}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
python - "$model_snapshot" <<'PY'
import json
import pathlib
import sys
import torch
import vllm

root = pathlib.Path(sys.argv[1])
assert vllm.__version__ == '0.31.0+cu129', vllm.__version__
assert torch.cuda.device_count() == 4, 'Expected four visible GPUs on this node'
for i in range(4):
    assert 'H100' in torch.cuda.get_device_name(i), torch.cuda.get_device_name(i)
config = json.loads((root / 'config.json').read_text())
assert config['num_hidden_layers'] == 48
assert config['num_experts'] == 128
assert config['num_experts_per_tok'] == 8
index = json.loads((root / 'model.safetensors.index.json').read_text())
shards = sorted(set(index['weight_map'].values()))
assert len(shards) == 16, len(shards)
for shard in shards:
    path = root / shard
    assert path.is_file() and path.stat().st_size > 0, f'Missing/empty shard: {path}'
print('Basic preflight passed; final runs additionally require archived checksums,')
print('source hashes, resolved configuration, network and correctness verification.')
PY
exec "${command[@]}"
