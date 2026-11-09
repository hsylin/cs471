"""Untimed diagnosis and fixed-map calibration on captured pilot inputs.

This is separate from the primary payback timers. Profiler measurements are
diagnostic, not policy speedups. A B2 observation prefix creates each resident
map; all timed calibration forwards use that fixed, verified map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from followup.router_tapes import load_router_bank
from skewbench.native import NativeLayer, Runtime, Weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rt = Runtime()
    torch = rt.torch
    weights = Weights(args.weights_dir)
    bank = load_router_bank(args.bank)
    if bank.metadata['split'] != 'pilot':
        raise ValueError('Profiling may use pilot documents only')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for batch in (128, 1024):
            for domain in (0, 1):
                indices = np.flatnonzero(bank.domain_ids == domain)[:batch]
                if len(indices) != batch:
                    raise ValueError('Insufficient captured records for calibration')
                hidden = torch.from_numpy(bank.hidden_states_bf16[indices].copy()).view(torch.bfloat16).to(rt.device)
                ids = torch.from_numpy(bank.expert_ids[indices].copy()).to(rt.device)
                combining = torch.from_numpy(bank.combining_weights[indices].copy()).to(rt.device)
                layer = NativeLayer(rt, weights, method='B2', replicas=32,
                                    window_steps=16, rebalance_interval=64,
                                    epsilon=.02, tau=.1, cooldown_steps=256, seed=0)
                # One real native update, followed by fixed-map warmup.
                for step in range(64):
                    layer.forward(hidden, ids, combining)
                    layer.step(step)
                layer.assert_maps_and_weights()
                correctness = layer.check_outputs(hidden, ids, combining)
                for _ in range(16):
                    layer.forward(hidden, ids, combining, observe=False)
                rt.barrier(); rt.sync()
                elapsed = []
                for _ in range(5):
                    rt.barrier(); rt.sync()
                    started = time.perf_counter()
                    for _ in range(32):
                        layer.forward(hidden, ids, combining, observe=False)
                    rt.sync()
                    elapsed.append(max(rt.gather_scalars(time.perf_counter()-started))/32)

                # Exact physical IDs from the executed native mapping kernel.
                ls = layer.layer_state
                ls.num_unpadded_tokens_tensors[0].fill_(batch)
                physical = layer.mapping_kernel(ids, ls.expert_load_view,
                    ls.logical_to_physical_map, ls.logical_replica_count,
                    torch.zeros((), dtype=torch.bool, device=rt.device),
                    ls.num_unpadded_tokens_tensors[0]).cpu().numpy()
                counts = np.bincount(physical.ravel(), minlength=160).reshape(2, 80)
                assert int(counts.sum()) == batch*8
                activities = [torch.profiler.ProfilerActivity.CPU,
                              torch.profiler.ProfilerActivity.CUDA]
                rt.barrier(); rt.sync()
                with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                    for _ in range(8):
                        with torch.profiler.record_function('captured_fixed_map_moe'):
                            layer.forward(hidden, ids, combining, observe=False)
                    rt.sync()
                stem = f'batch{batch}-domain{domain}-rank{rt.rank}'
                prof.export_chrome_trace(str(args.output_dir/f'{stem}.json'))
                (args.output_dir/f'{stem}.txt').write_text(
                    prof.key_averages().table(sort_by='self_device_time_total', row_limit=40))
                rows.append({'batch_tokens':batch, 'domain':domain,
                    'fixed_map':layer.map_snapshot().tolist(),
                    'completed_seconds_per_forward':elapsed,
                    'physical_tokens_per_slot':counts.tolist(),
                    'tokens_by_rank':counts.sum(axis=1).tolist(),
                    'active_physical_experts_by_rank':(counts>0).sum(axis=1).tolist(),
                    'profile_forward_count':8, 'correctness':correctness})
                layer.close()
                del layer, hidden, ids, combining
                torch.cuda.empty_cache()
        if rt.rank == 0:
            (args.output_dir/'calibration.json').write_text(json.dumps({
                'scope':'pilot fixed-map kernel diagnosis, not policy or payback measurement',
                'capture_sha256':bank.sha256, 'runtime':rt.metadata,
                'profiler':'torch.profiler CPU+CUDA; profiled runs excluded from calibration timings',
                'calibration_trials_per_cell':5, 'forwards_per_trial':32,
                'rows':rows}, indent=2)+'\n')
    finally:
        rt.close()


if __name__ == '__main__':
    main()
