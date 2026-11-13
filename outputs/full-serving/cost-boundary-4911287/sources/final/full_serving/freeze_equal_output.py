"""Freeze six new fixed-length sessions using prior OFF lengths, not speeds."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from full_serving import custom_compare as compare
from full_serving.output_lengths import reference_targets


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(parent_path, reference_run, output):
    parent_path, reference_run, output = map(lambda p: Path(p).resolve(), (parent_path, reference_run, output))
    parent = compare.load_admitted_protocol(parent_path)
    compare.require(parent["schema"] == compare.COMPACT_SCHEMA and not parent.get("equal_output_length"),
                    "completed natural-EOS two-workload parent is required")
    compare.require((reference_run / "frozen-protocol.json").read_bytes() == parent_path.read_bytes(), "reference protocol differs")
    status = json.loads((reference_run / "status.json").read_bytes())
    compare.require(status.get("state") == "comparison_completed"
                    and status.get("completed_sessions") == [j["session_id"] for j in compare.schedule(parent)],
                    "all original reference sessions must be completed")
    compare.require(not output.parent.exists(), "new fixed-length freeze directory is required")
    shutil.copytree(parent_path.parent, output.parent)
    protocol = copy.deepcopy(parent)
    protocol["equal_output_length"] = True
    protocol["generation"]["ignore_eos"] = True
    for dataset in protocol["datasets"]:
        name = dataset["dataset"]
        block = dataset["blocks"][0]
        block["arm_order"] = list(reversed(block["arm_order"]))
        dataset["arm_position_counts"] = {a: [int(block["arm_order"][i] == a) for i in range(3)] for a in compare.ARMS}
        cap = protocol["generation_by_dataset"][name]["max_tokens"]
        for label, item, phase in (("warmup", dataset["warmup"], "warmup-00"), ("measured", block, "measured")):
            reference = reference_run / name / block["block_id"] / "off" / (phase + ".jsonl")
            raw = reference.read_bytes()
            targets = reference_targets(raw, item, cap)
            item["output_reference_file"] = f"output-length-reference/{name}-{label}.jsonl"
            path = output.parent / item["output_reference_file"]
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            item["output_reference_sha256"] = hashlib.sha256(raw).hexdigest()
            item["output_lengths_file"] = f"output-lengths/{name}-{label}.json"
            item["output_lengths_sha256"] = write(output.parent / item["output_lengths_file"], targets)
    protocol["equal_length_reference"] = {
        "natural_protocol_sha256": compare.native.sha(parent_path),
        "rule": "Per-request target equals the previous natural-EOS OFF completion count. No speed-based selection. Pilot warmup00 counts are reused for both warmup passes; measured counts bind each of the same16 prompts.",
        "request_settings": "Each request sends max_tokens=N, min_tokens=N, ignore_eos=true, stop=[], stop_token_ids=[]. All arms charge this common sampling/control overhead.",
        "meaning": "Equal generated token count per prompt, not identical token sequence, expert routing, batching or mathematical answer quality.",
        "freshness": "Fresh server executions of previously used content and unchanged calendars; not fresh payloads or an independent generalization sample."}
    protocol["analysis"]["generation_caveat"] = protocol["equal_length_reference"]["meaning"]
    protocol["execution"]["arm_order_rule"] = "Reverse the previous fixed arm order in each workload before execution; no outcome-based order selection."
    protocol["execution"]["global_arm_position_counts"] = {a: [sum(d["blocks"][0]["arm_order"][i] == a for d in protocol["datasets"])
                                                                for i in range(3)] for a in compare.ARMS}
    protocol["acceptance"]["correctness_required"].append("Every measured and warmup request completes exactly its frozen reference output-token count with length termination; reject early stopping and failed requests.")
    protocol["amendment"] = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "User requested a fair equal-output-length comparison after the natural-EOS results.",
        "parent_natural_protocol_sha256": compare.native.sha(parent_path),
        "prior_measurements_preserved": True, "private_payloads_read": False,
        "same_prompts_arrivals_model_hardware_and_method": True,
        "statistical_scope": "One group per workload, three arms; descriptive only, no confidence interval or reproducibility claim."}
    protocol["freeze"] = {"created_utc": protocol["amendment"]["created_utc"],
        "no_fixed_length_model_execution_performed": True, "previous_natural_payload_execution_disclosed": True}
    protocol["protocol_sha256"] = compare.canonical_hash({k: v for k, v in protocol.items() if k != "protocol_sha256"})
    write(output, protocol)
    compare.load_admitted_protocol(output)
    compare.load_source_metadata(protocol, output)
    return {"status": "frozen", "protocol_sha256": compare.native.sha(output), "sessions": compare.schedule(protocol),
            "measured_output_tokens_per_arm": {d["dataset"]: sum(json.loads((output.parent / d["blocks"][0]["output_lengths_file"]).read_bytes()).values())
                                               for d in protocol["datasets"]}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("parent", "reference-run", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args(argv)
    print(json.dumps(freeze(args.parent, args.reference_run, args.output), indent=2))


if __name__ == "__main__":
    main()
