#!/usr/bin/env python3
"""Descriptive DP-versus-greedy assignment comparison on old pilot inputs only.

No GPU work, timing inference, policy selection, threshold search or final input
inspection. The exact 48 deterministic indices per development cell are checked
against the earlier saved input artifact before computing any comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from followup.router_tapes import load_router_bank
from optimization.common import source_fingerprint
from optimization.tuning import input_indices
from resident_dispatch.assignments import generate_assignment
from resident_dispatch.online import cpu_greedy_assignment, POLICY_ID


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype=np.int64).tobytes()).hexdigest()


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "min": float(x.min()), "max": float(x.max())}


def measures(physical, block_m):
    counts = np.bincount(np.asarray(physical).reshape(-1), minlength=160).reshape(2,80)
    return {"per_rank_tokens": counts.sum(1).tolist(),
            "per_rank_active_copies": np.count_nonzero(counts, axis=1).tolist(),
            "per_rank_padded_rows": (((counts+block_m-1)//block_m)*block_m).sum(1).tolist()}


def summarize(rows):
    return {
        "unique_inputs": len(rows),
        "greater_token_peak_inputs": sum(r["max_token_delta"] > 0 for r in rows),
        "equal_token_peak_inputs": sum(r["max_token_delta"] == 0 for r in rows),
        "identical_physical_assignment_inputs": sum(r["changed_route_fraction"] == 0 for r in rows),
        "dp_max_tokens": stats([max(r["dp"]["per_rank_tokens"]) for r in rows]),
        "greedy_max_tokens": stats([max(r["greedy"]["per_rank_tokens"]) for r in rows]),
        "greedy_minus_dp_max_tokens": stats([r["max_token_delta"] for r in rows]),
        "greedy_over_dp_max_tokens": stats([r["max_token_ratio"] for r in rows]),
        "changed_route_fraction": stats([r["changed_route_fraction"] for r in rows]),
        "max_padded_rows_delta": stats([r["max_padded_rows_delta"] for r in rows]),
        "greedy_over_dp_max_padded_rows": stats([r["max_padded_rows_ratio"] for r in rows]),
        "greedy_lower_padded_peak_inputs": sum(r["max_padded_rows_delta"] < 0 for r in rows),
        "greedy_equal_padded_peak_inputs": sum(r["max_padded_rows_delta"] == 0 for r in rows),
        "greedy_higher_padded_peak_inputs": sum(r["max_padded_rows_delta"] > 0 for r in rows),
        "max_active_copy_count_equal_all_inputs": all(r["max_active_copies_delta"] == 0 for r in rows),
        "total_active_copy_count_equal_all_inputs": all(r["total_active_copies_delta"] == 0 for r in rows),
        "total_padded_work_equal_all_inputs": all(r["total_padded_rows_delta"] == 0 for r in rows),
        "per_rank_token_delta": [stats([r["per_rank_token_delta"][i] for r in rows]) for i in range(2)],
        "per_rank_active_copy_delta": [stats([r["per_rank_active_copy_delta"][i] for r in rows]) for i in range(2)],
        "per_rank_padded_rows_delta": [stats([r["per_rank_padded_rows_delta"][i] for r in rows]) for i in range(2)],
    }


def analyze(manifest_path, banks_dir, pilot_inputs):
    manifest = json.loads(manifest_path.read_text())
    digest = canonical({k:v for k,v in manifest.items() if k != "freeze"})
    if manifest.get("freeze",{}).get("sha256") != digest:
        raise ValueError("Manifest is not frozen or changed")
    expected_blocks = ["opt-pilot-01", "opt-pilot-02"]
    if (manifest["phase"] != "validation" or manifest["split"] != "pilot"
            or manifest["matrix"]["block_ids"] != expected_blocks
            or manifest["matrix"]["regimes"] != ["domain0","domain1"]
            or manifest["matrix"]["batch_tokens"] != [128,1024]):
        raise ValueError("Analysis is restricted to the two existing development banks")
    sources = {p:sha(ROOT/p) for p in ("resident_dispatch/assignments.py",
               "resident_dispatch/online.py", "optimization/tuning.py")}
    for name in ("resident_dispatch/assignments.py", "resident_dispatch/online.py"):
        if sources[name] != manifest["execution_source_sha256"][name]:
            raise ValueError(f"Frozen assignment source differs: {name}")
    if source_fingerprint()[0] != manifest["environment"]["optimization_sha256"]:
        raise ValueError("Frozen deterministic input generator source differs")
    mapping = np.r_[np.arange(128),np.arange(32)]
    cells, artifacts, all_rows = [], [], []
    for block in expected_blocks:
        entry = next(e for e in manifest["banks"] if e["block_id"] == block)
        path = banks_dir/entry["capture_path"]
        bank = load_router_bank(path)
        if bank.sha256 != entry["capture_sha256"] or bank.metadata["split"] != "pilot":
            raise ValueError("Old development capture identity differs")
        if bank.metadata["correctness_pass"] is not True:
            raise ValueError("Capture lacks correctness attestation")
        for domain in (0,1):
            for batch in (128,1024):
                indices,seed = input_indices(bank.domain_ids,block,domain,batch)
                logical = bank.expert_ids[indices]
                previous_path = pilot_inputs/f"{block}-d{domain}-b{batch}.npz"
                with np.load(previous_path,allow_pickle=False) as previous:
                    if (not np.array_equal(previous["indices"],indices)
                            or not np.array_equal(previous["logical_ids"],logical)
                            or not np.array_equal(previous["physical_to_logical_map"],mapping)):
                        raise AssertionError("Generated inputs differ from the old development artifact")
                    saved_dp = previous["one_copy_home_physical_ids"].copy()
                    prior_metadata = json.loads(str(previous["metadata"]))
                    if prior_metadata["capture_sha256"] != bank.sha256:
                        raise AssertionError("Saved development inputs came from another bank")
                block_m = manifest["default_configs"][str(batch)]["BLOCK_SIZE_M"]
                rows = []
                for raw,ids in enumerate(logical):
                    dp = generate_assignment(ids,mapping,method="one_copy_home",block_size_m=block_m)
                    if not np.array_equal(dp.physical_ids,saved_dp[raw]):
                        raise AssertionError("Offline DP assignment differs from the saved development artifact")
                    greedy = cpu_greedy_assignment(ids,mapping)
                    if not np.array_equal(mapping[greedy.physical_ids],ids):
                        raise AssertionError("Greedy assignment changed a logical route")
                    a,b = measures(dp.physical_ids,block_m),measures(greedy.physical_ids,block_m)
                    delta = max(b["per_rank_tokens"])-max(a["per_rank_tokens"])
                    if delta < 0:
                        raise AssertionError("Greedy improved on the claimed exact token tie-break optimum")
                    row = {"block_id":block,"domain":domain,"batch_tokens":batch,
                           "raw_index":raw,"input_phase":"conditioning" if raw<16 else "measured",
                           "block_size_m":block_m,"dp":a,"greedy":b,
                           "max_token_delta":delta,
                           "max_token_ratio":max(b["per_rank_tokens"])/max(a["per_rank_tokens"]),
                           "max_active_copies_delta":max(b["per_rank_active_copies"])-max(a["per_rank_active_copies"]),
                           "total_active_copies_delta":sum(b["per_rank_active_copies"])-sum(a["per_rank_active_copies"]),
                           "total_padded_rows_delta":sum(b["per_rank_padded_rows"])-sum(a["per_rank_padded_rows"]),
                           "max_padded_rows_delta":max(b["per_rank_padded_rows"])-max(a["per_rank_padded_rows"]),
                           "max_padded_rows_ratio":max(b["per_rank_padded_rows"])/max(a["per_rank_padded_rows"]),
                           "changed_route_fraction":float(np.mean(dp.physical_ids!=greedy.physical_ids)),
                           "per_rank_token_delta":(np.array(b["per_rank_tokens"])-a["per_rank_tokens"]).tolist(),
                           "per_rank_active_copy_delta":(np.array(b["per_rank_active_copies"])-a["per_rank_active_copies"]).tolist(),
                           "per_rank_padded_rows_delta":(np.array(b["per_rank_padded_rows"])-a["per_rank_padded_rows"]).tolist(),
                           "dp_physical_ids_sha256":array_hash(dp.physical_ids),
                           "greedy_physical_ids_sha256":array_hash(greedy.physical_ids)}
                    assert row["max_active_copies_delta"] == row["total_active_copies_delta"] == row["total_padded_rows_delta"] == 0
                    rows.append(row)
                cells.append({"block_id":block,"domain":domain,"batch_tokens":batch,
                              "block_size_m":block_m,"input_seed":seed,
                              "indices_sha256":array_hash(indices),
                              "all48":summarize(rows),"measured32":summarize(rows[16:]),
                              "conditioning16":summarize(rows[:16]),"inputs":rows})
                artifacts.append({"block_id":block,"domain":domain,"batch_tokens":batch,
                    "capture_path":str(path),"capture_sha256":bank.sha256,
                    "pilot_input_path":str(previous_path),"pilot_input_sha256":sha(previous_path)})
                all_rows.extend(rows)
    measured = [r for r in all_rows if r["input_phase"] == "measured"]
    return {"schema_version":1,"analysis_kind":"descriptive_current_batch_assignment_approximation",
            "measurement_kind":"cpu_count_diagnostics_only","split":"old_development_only",
            "policy_id":POLICY_ID,"manifest_path":str(manifest_path),"manifest_sha256":digest,
            "source_sha256":sources,"analysis_script_sha256":sha(Path(__file__)),
            "source_blocks":2,"cells":cells,"artifacts":artifacts,
            "all_unique_inputs":summarize(all_rows),"measured_unique_inputs":summarize(measured),
            "measured_by_batch":{str(b):summarize([r for r in measured if r["batch_tokens"]==b]) for b in (128,1024)},
            "scope_notes":["Exactly 48 original indices per cell:16 conditioning and32 measured, not fresh final demand.",
                "The eight cyclic repetitions do not create new unique inputs or independent samples.",
                "DP minimizes token peak among one-home assignments attaining the minimum active-copy peak; greedy retains the active-count objective with approximate token tie-breaking.",
                "Total active copies and total padded rows are invariant between these one-home assignments; rank distribution can differ.",
                "Token/padding differences are descriptive mechanism quantities, not latency estimates or runtime bounds.",
                "No GPU timing, confidence interval, selector tuning, threshold search or final-trace analysis."]}


def markdown(result):
    s=result["measured_unique_inputs"]
    all_inputs=result["all_unique_inputs"]
    lines=["# Resident assignment approximation on old development inputs", "",
        f"Across all {all_inputs['unique_inputs']} original input batches, greedy adds a mean {all_inputs['greedy_minus_dp_max_tokens']['mean']:.3f} routed tokens to the heavier rank relative to the offline DP tie-break optimum; the mean greedy/DP token-peak ratio is {all_inputs['greedy_over_dp_max_tokens']['mean']:.6f}. This includes the 16 conditioning and 32 measured inputs in each of eight cells.", "",
        f"Across the {s['unique_inputs']} unique measured inputs, greedy token-peak load exceeds the offline DP tie-break optimum on {s['greater_token_peak_inputs']} inputs and equals it on {s['equal_token_peak_inputs']}. The mean excess is {s['greedy_minus_dp_max_tokens']['mean']:.3f} routed tokens; the maximum is {s['greedy_minus_dp_max_tokens']['max']:.0f}. The mean greedy/DP token-peak ratio is {s['greedy_over_dp_max_tokens']['mean']:.6f}, with maximum {s['greedy_over_dp_max_tokens']['max']:.6f}.", "",
        "Both assignments attain the same exact minimum maximum-active-copy count on every input. Total active copies and total padded rows are also identical. This does not make their runtimes equal: the assignment of token and padded work across ranks differs, and this CPU comparison does not measure the GPU selector's cost or completed layer latency.", "",
        "| Old block | Domain | Batch | Unique measured inputs | Mean excess peak tokens | Maximum excess | Mean token-peak ratio | Mean padded-peak difference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in result["cells"]:
        x=c["measured32"]
        lines.append(f"| {c['block_id']} | {c['domain']} | {c['batch_tokens']} | 32 | {x['greedy_minus_dp_max_tokens']['mean']:.3f} | {x['greedy_minus_dp_max_tokens']['max']:.0f} | {x['greedy_over_dp_max_tokens']['mean']:.6f} | {x['max_padded_rows_delta']['mean']:+.1f} |")
    lines += ["",f"Greedy's maximum per-rank padded work is lower on {s['greedy_lower_padded_peak_inputs']} measured inputs, equal on {s['greedy_equal_padded_peak_inputs']}, and higher on {s['greedy_higher_padded_peak_inputs']}. DP optimizes its token-count tie-break, not padded work. The mean changed-route fraction is {s['changed_route_fraction']['mean']:.3%}; equal summary counts therefore need not imply identical physical assignments.", "",
        "The JSON includes all 384 input-level records, both ranks' token/active/padded values and differences, and source/input hashes. Inputs were regenerated with the exact original input_indices function and checked against all eight existing pilot NPZ artifacts; every regenerated DP assignment matches the saved assignment. No new demand was selected, and no frozen execution source was changed.", "",
        "These are descriptive comparisons over two reused source-document blocks. The eight cycles of each measured input are not independent observations. Neither a token gap nor an unchanged active-copy optimum establishes the sign or size of a runtime gap. The fully charged GPU validation remains the performance test; this result does not authorize changing the selector.", "",
        "Reproduction:", "", "```bash",
        "python scripts/analyze_resident_approximation.py \\",
        "  --manifest configs/resident-online-validation-frozen-v1.json \\",
        "  --banks-dir /path/to/optimization-capture \\",
        "  --pilot-inputs /path/to/resident-results/pilot/inputs \\",
        "  --output /fresh/path/approximation.json \\",
        "  --markdown /fresh/path/approximation.md", "```", ""]
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--banks-dir',type=Path,required=True)
    p.add_argument('--pilot-inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--markdown',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.markdown.exists(): raise FileExistsError('Use fresh output paths')
    result=analyze(a.manifest,a.banks_dir,a.pilot_inputs)
    for path in (a.output,a.markdown):path.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    with a.markdown.open('x') as f:f.write(markdown(result))
    print(json.dumps(result['measured_unique_inputs'],sort_keys=True))


if __name__=='__main__':main()
