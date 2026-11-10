"""Freeze the explicitly requested two-workload, one-group, three-arm study.

Reuse the first reserved group and existing arrival calendar, without inspecting
private text or performance outcomes. Keep the parent protocol as evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from full_serving import custom_compare as compare


def amend(parent_path, output):
    parent_path, output = Path(parent_path).resolve(), Path(output).resolve()
    parent = compare.load_admitted_protocol(parent_path)
    compare.require(parent["schema"] == "resident_compare_protocol_v1", "the original full protocol is required")
    compare.require(not output.parent.exists(), "amendment destination must be new")
    # All copied files are public metadata; unchanged parent paths remain valid.
    shutil.copytree(parent_path.parent, output.parent)
    parent_copy = output.parent / "parent-protocol.json"
    parent_copy.write_bytes(parent_path.read_bytes())
    protocol = copy.deepcopy(parent)
    protocol.update(schema=compare.COMPACT_SCHEMA, status="frozen")
    protocol["source_admission"].update(parent_protocol_file="parent-protocol.json",
                                         parent_protocol_sha256=compare.native.sha(parent_copy))
    protocol["datasets"] = [copy.deepcopy(next(d for d in parent["datasets"] if d["dataset"] == name))
                            for name in compare.COMPACT_DATASETS]
    for index, dataset in enumerate(protocol["datasets"]):
        dataset["blocks"] = dataset["blocks"][:1]
        order = ["custom", "off", "on"] if index == 0 else ["off", "on", "custom"]
        dataset["blocks"][0]["arm_order"] = order
        dataset["arm_position_counts"] = {arm: [int(order[p] == arm) for p in range(3)] for arm in compare.ARMS}
    protocol["generation_by_dataset"] = {"longbench": {"max_tokens": 16}, "numinamath": {"max_tokens": 2048}}
    protocol["analysis"].update(
        experimental_unit="One reserved sixteen-request content group and its existing arrival calendar per dataset.",
        estimand="Per-dataset paired treatment/baseline response ratios for this one group; no pooling.",
        uncertainty="Descriptive only: one group cannot estimate between-group or repeated-run variation.",
        formal_hypothesis_tests=False, inferential_intervals=False,
        multiplicity="No tests or confidence intervals; point margins are descriptive, not acceptance guarantees.",
        generation_caveat="LongBench is deliberately capped at16 output tokens to target prefill; NuminaMath uses2048. Full prompts are unchanged. These generation-budget workloads are not semantic-quality benchmarks.")
    protocol["scope"].update(main_content_sources=2, final_distinct_requests=32,
        measured_http_executions=96, fresh_server_launches=6, warmup_http_executions=96,
        total_planned_warmup_and_measured_http_executions=192,
        scope_limit="Two descriptive short full-model episodes, one group each; no repeated-run uncertainty or production-capacity claim.")
    protocol["execution"].update(estimated_budget_hours=[1, 2],
        budget_note="Six fresh server launches; previous startups take several minutes each. No shortening admitted work.",
        arm_order_rule="LongBench: custom,off,on; NuminaMath: off,on,custom. Each arm occupies different positions across workloads; no within-workload replication.",
        global_arm_position_counts={arm: [sum(d["blocks"][0]["arm_order"][i] == arm for d in protocol["datasets"])
                                         for i in range(3)] for arm in compare.ARMS})
    protocol["workload_phase_check"] = {
        "intended": {"longbench": "prefill", "numinamath": "decode"},
        "rule": "Classify each arm by its larger summed native request prefill/decode duration, with16 completed counts in each phase. No outcome-based retuning or replacement.",
        "limit": "Request-phase dominance is not proof of GPU compute/bandwidth boundedness. Report any failure to achieve intended phase dominance.",
        "selection": "First reserved group of each dataset, unchanged full prompts/calendar/rate; dataset and output-budget choice follows the user's phase-focused scope reduction, not final speed rankings."}
    protocol["amendment"] = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "User explicitly requested only two datasets, one prefill-bound and one decode-bound, one group, three methods.",
        "parent_protocol_sha256": compare.native.sha(parent_path),
        "preserved_parent_measurements": "The superseded run completed four WildChat sessions before cancellation; neither selected final dataset ran there.",
        "statistical_change": "Four-block intervals removed, not reused for the one-group design.",
        "private_payloads_read": False, "new_model_execution_performed": False}
    protocol["freeze"] = {"created_utc": protocol["amendment"]["created_utc"],
        "no_selected_final_model_execution_performed": True, "no_final_selection_or_rate_tuning": True}
    protocol["protocol_sha256"] = compare.canonical_hash({k: v for k, v in protocol.items() if k != "protocol_sha256"})
    compare.validate_protocol(protocol)
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    compare.load_source_metadata(protocol, output)
    compare.validate_admission_artifacts(protocol, output)
    return {"status": "frozen", "protocol_sha256": compare.native.sha(output),
            "sessions": compare.schedule(protocol), "private_payloads_read": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(amend(args.parent, args.output), indent=2))


if __name__ == "__main__":
    main()
