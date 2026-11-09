#!/usr/bin/env python3
"""Compose v2 replay into a new directory without modifying captured banks/v1.

This is deterministic input construction, not performance analysis. The single
transient occupies measured120..135; the fixed proposal follows measured127.
All banks retain their original capture script hash. This script's hash records
the later composition operation separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from followup.router_tapes import compose_router_tape, file_sha256, load_router_bank


def compose_directory(banks_dir: Path, output_dir: Path, split: str = "all") -> dict:
    banks_dir, output_dir = Path(banks_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a new empty tape directory; v1/v2 artifacts are never overwritten")
    if output_dir.resolve() == banks_dir.resolve():
        raise ValueError("Replay output must differ from captured-bank directory")
    banks = []
    seen = set()
    for path in sorted(banks_dir.glob("*.npz")):
        bank = load_router_bank(path)
        if split != "all" and bank.metadata["split"] != split:
            continue
        if bank.metadata["block_id"] in seen:
            raise ValueError("Duplicate source block")
        seen.add(bank.metadata["block_id"])
        banks.append(bank)
    if not banks:
        raise ValueError("No matching captured source banks")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bank in banks:
        before = bank.sha256
        for batch in (128, 1024):
            for regime in ("stable", "transient", "persistent", "switching"):
                path = output_dir/f"{bank.metadata['block_id']}-{regime}-b{batch}.npz"
                tape = compose_router_tape(bank.path, path, regime=regime, batch_tokens=batch,
                    num_steps=672, warmup_steps=32, phase_steps=112 if regime == "switching" else 120,
                    duration_steps=16, repeat_steps=1024, seed=471)
                schedule = tape.metadata["domain_schedule"]
                if regime == "transient" and (sum(schedule) != 16 or schedule[152:168] != [1]*16):
                    raise AssertionError("Transient is not exactly one declared event")
                rows.append({"block_id":bank.metadata["block_id"], "split":bank.metadata["split"],
                             "regime":regime,"batch_tokens":batch,"path":str(path),
                             "sha256":tape.sha256,"capture_sha256":bank.sha256})
        if file_sha256(bank.path) != before:
            raise AssertionError("Capture bank changed during composition")
        print(json.dumps({"event":"block_composed","block_id":bank.metadata["block_id"],
                          "replays":8,"capture_sha256":before}),flush=True)
    result = {"schema":"qwen_router_replay_index_v2", "composition_script_sha256":file_sha256(__file__),
              "router_tapes_module_sha256":file_sha256(Path(__file__).resolve().parents[1]/"followup/router_tapes.py"),
              "contract":{"raw_steps":672,"warmup_steps":32,"decision_step":127,
                          "phase_steps":120,"switching_phase_steps":112,"duration_steps":16,
                          "repeat_steps":1024,"seed":471,"natural_temporal_demand":False},
              "capture_provenance":"Banks and original capture-script hashes retained; no recapture or mutation",
              "rows":rows}
    (output_dir/"replay-index.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--split",choices=("pilot","final","all"),default="all")
    a=p.parse_args()
    result=compose_directory(a.banks_dir,a.output_dir,a.split)
    print(json.dumps({"event":"composition_complete","replays":len(result["rows"]),
                      "index":str(a.output_dir/"replay-index.json")}),flush=True)


if __name__=="__main__":main()
