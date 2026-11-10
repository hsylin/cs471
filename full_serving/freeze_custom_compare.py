"""Validate and, only after pilot admission, freeze the held-out comparison.

``check`` and ``freeze`` without ``--execute`` read public evidence only.
Execution materializes unchanged admitted rows; it performs no model inference.
Both output directories must be fresh. Private text never enters public files.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import tempfile

from full_serving import custom_compare as compare
from full_serving.custom_inputs import load_source_metadata, verify_private_request
from full_serving.data import digest, file_digest
from full_serving.prepare_arrivals import digest as arrival_digest
from full_serving.resident_plugin import validate_config, verify_source_manifest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def check_draft(path):
    """Public-only verification, including independent calendar reconstruction."""
    path = Path(path).resolve()
    protocol = read_json(path)
    require(protocol.get("status") == "draft" and protocol.get("final_execution_allowed") is False,
            "expected a nonexecutable draft")
    require(protocol.get("protocol_sha256") == digest({k: v for k, v in protocol.items() if k != "protocol_sha256"}),
            "draft protocol seal differs")
    compare.validate_protocol(protocol, allow_draft=True)
    metadata = load_source_metadata(protocol, path)
    source = protocol["source_admission"]
    arrivals_path = compare.artifact_path(path, source["arrival_manifest_file"])
    require(file_digest(arrivals_path) == source["arrival_manifest_sha256"], "arrival manifest bytes changed")
    arrivals = read_json(arrivals_path)
    require(arrivals.get("content_sha256") == arrival_digest({k: v for k, v in arrivals.items() if k != "content_sha256"}),
            "arrival manifest seal differs")
    parent_blocks = {b["id"]: b for b in arrivals["blocks"]}
    records = []
    for dataset in protocol["datasets"]:
        pilot_path = compare.artifact_path(path, dataset["pilot_calendar_file"])
        require(file_digest(pilot_path) == dataset["pilot_calendar_sha256"], "original pilot calendar bytes changed")
        pilot = read_json(pilot_path)
        rate = pilot["offered_requests_per_s"]
        require(rate == dataset["arrival_rate_requests_per_s"] and rate > 0, "rate differs from original pilot")
        dates = set()
        kinds = []
        for block in dataset["blocks"]:
            calendar_path = compare.artifact_path(path, block["calendar_file"])
            require(file_digest(calendar_path) == block["calendar_sha256"], "draft calendar bytes changed")
            calendar = read_json(calendar_path)
            body = calendar["source_block"]
            parent = parent_blocks[body["parent_block_id"]]
            require(parent["split"] == "final" and body["parent_arrival_manifest_file_sha256"] == file_digest(arrivals_path),
                    "calendar parent is not the reserved final arrival source")
            require(body["source_sha256"] == parent["source_sha256"]
                    and body["trace"] == parent["trace"] and body["source_date_utc"] == parent["source_date_utc"]
                    and body["source_origin_ns"] == parent["source_origin_ns"], "calendar source identity differs")
            for key in ("row_indices", "raw_timestamps", "timestamp_ns", "context_tokens", "generated_tokens"):
                require(body[key] == parent[key][:16], f"calendar source prefix differs: {key}")
            offsets, horizon = parent["original_offsets_ns"][:16], parent["original_offsets_ns"][16]
            require(horizon > offsets[-1] and body["original_offsets_ns"] == offsets
                    and body["original_horizon_ns"] == horizon and body["jobs"] == 16
                    and body["row_start"] == parent["row_indices"][0]
                    and body["row_stop_exclusive"] == body["boundary_row"] == parent["row_indices"][16]
                    and body["boundary_timestamp"] == parent["raw_timestamps"][16]
                    and body["boundary_timestamp_ns"] == parent["timestamp_ns"][16]
                    and body["boundary_context_tokens"] == parent["context_tokens"][16]
                    and body["boundary_generated_tokens"] == parent["generated_tokens"][16],
                    "calendar source horizon or boundary differs")
            target = Fraction(16*10**9, 1)/Fraction(str(rate))
            scale = target/horizon
            require(calendar["request_ids"] == block["request_ids"]
                    and calendar["absolute_arrivals"] is True and calendar["source_request_sizes_executed"] is False
                    and calendar["offered_requests_per_s"] == rate
                    and calendar["arrival_offsets_ns"] == [int(x*scale) for x in offsets]
                    and calendar["horizon_ns"] == int(target)
                    and calendar["scale_numerator"] == scale.numerator and calendar["scale_denominator"] == scale.denominator,
                    "calendar differs from frozen IDs/rate/exact rational scaling")
            selection = calendar["rate_selection"]
            require(selection["pilot_calendar_sha256"] == dataset["pilot_calendar_sha256"]
                    and selection["calibration_file_sha256"] == pilot["rate_selection"]["calibration_file_sha256"],
                    "calendar rate-admission evidence differs")
            dates.add(parent["source_date_utc"])
            kinds.append(parent["trace"])
            records.append({"dataset": dataset["dataset"], "block_id": block["block_id"],
                            "calendar_sha256": block["calendar_sha256"], "request_ids_sha256": block["request_ids_sha256"],
                            "source_parent_block": parent["id"], "offered_requests_per_s": rate})
        require(len(dates) == 4 and sorted(kinds) == ["code", "code", "conversation", "conversation"],
                "expected four separate dates, two code and two conversation")
    require(len(records) == 20 and len(compare.schedule(protocol, allow_draft=True)) == 60,
            "bounded comparison size differs")
    return protocol, metadata, {"status": "passed", "draft_file_sha256": file_digest(path),
        "calendars": records, "private_payloads_read": False, "datasets": 5,
        "measured_requests": 960, "server_launches": 60, "warmup_executions": 960}


def bind_pilot(protocol, draft_path, config_path, source_path, admission_path):
    """Validate actual public pilot/configuration bindings before private access."""
    bound = copy.deepcopy(protocol)
    config = validate_config(read_json(config_path))
    require(config["mode"] == "resident_v1" and config.get("startup_diagnostic") is False,
            "final comparison needs the admitted timed resident configuration")
    verify_source_manifest(read_json(source_path))
    bound["custom"].update(entrypoint="full_serving.custom_entrypoint", mode="resident_v1",
        bootstrap_path="full_serving/bootstrap", config_file=str(Path(config_path).resolve()),
        config_sha256=file_digest(config_path), source_manifest_file=str(Path(source_path).resolve()),
        source_manifest_sha256=file_digest(source_path))
    bound["pilot_admission"] = {"file": str(Path(admission_path).resolve()),
                               "sha256": file_digest(admission_path), "status": "passed"}
    compare.validate_admission_artifacts(bound, draft_path)
    return bound


def _source_rows(path, expected_hash):
    raw = Path(path).read_bytes()
    require(digest(raw) == expected_hash and raw.endswith(b"\n"), "admitted source payload bytes changed")
    lines = raw.split(b"\n")[:-1]
    require(all(lines), "blank line in admitted private source")
    return raw, [json.loads(line) for line in lines]


def _private_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)


def freeze(draft_path, config_path, source_path, admission_path, suite_private,
           private_output, output, *, execute=False):
    draft_path, output, private_output = Path(draft_path).resolve(), Path(output).resolve(), Path(private_output).resolve()
    protocol, metadata, check = check_draft(draft_path)
    bound = bind_pilot(protocol, draft_path, config_path, source_path, admission_path)
    require(not output.parent.exists() and not private_output.exists(), "public/private output directories must be fresh")
    if not execute:
        return {**check, "pilot_admission": "passed", "execute": False,
                "prospective_output": str(output), "private_payloads_read": False}
    # No private path is opened before every preceding public admission gate.
    output.parent.parent.mkdir(parents=True, exist_ok=True)
    private_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    public_stage = Path(tempfile.mkdtemp(prefix=".freeze-public-", dir=output.parent.parent))
    private_stage = Path(tempfile.mkdtemp(prefix=".freeze-private-", dir=private_output.parent))
    evidence = []
    try:
        for dataset in bound["datasets"]:
            name = dataset["dataset"]
            final_raw, finals = _source_rows(Path(suite_private)/name/"final.jsonl", dataset["reserved_source_final_file_sha256"])
            warm_raw, warmups = _source_rows(Path(suite_private)/name/"smoke.jsonl", dataset["warmup"]["requests_sha256"])
            expected = metadata["datasets"][name]
            require([r["id"] for r in finals] == expected["final_ids"]
                    and [r["id"] for r in warmups] == expected["warmup_ids"], "source row order differs from public admission")
            for row in finals+warmups:
                verify_private_request(row, metadata["by_id"][row["id"]])
            _private_write(compare.private_path(private_stage, dataset["warmup"]["requests_file"]), warm_raw)
            for index, block in enumerate(dataset["blocks"]):
                rows = finals[index*16:(index+1)*16]
                # Preserve every source field and string; only JSONL byte layout
                # is canonicalized, with no token/character truncation.
                raw = ("".join(json.dumps(r, ensure_ascii=True, sort_keys=True)+"\n" for r in rows)).encode()
                _private_write(compare.private_path(private_stage, block["requests_file"]), raw)
                block["requests_sha256"] = digest(raw)
                evidence.append({"dataset": name, "block_id": block["block_id"], "requests": 16,
                    "source_final_file_sha256": digest(final_raw), "requests_file": block["requests_file"],
                    "requests_sha256": digest(raw), "ordered_ids_sha256": digest([r["id"] for r in rows])})

        def copy_public(original, relative):
            target = public_stage/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(original).read_bytes())
            return relative
        for stem in ("config", "source_manifest"):
            key = stem+"_file"
            bound["custom"][key] = copy_public(bound["custom"][key], "implementation/"+stem+".json")
        bound["pilot_admission"]["file"] = copy_public(bound["pilot_admission"]["file"], "implementation/pilot-admission.json")
        for key, value in list(bound["source_admission"].items()):
            if key.endswith("_file"):
                bound["source_admission"][key] = copy_public(compare.artifact_path(draft_path, value),
                    "source-admission/"+key.removesuffix("_file")+".json")
                require(file_digest(public_stage/bound["source_admission"][key])
                        == bound["source_admission"][key.removesuffix("_file")+"_sha256"],
                        "public source evidence changed while freezing")
        for dataset in bound["datasets"]:
            dataset["pilot_calendar_file"] = copy_public(compare.artifact_path(draft_path, dataset["pilot_calendar_file"]),
                                                        "pilot-calendars/"+dataset["dataset"]+".json")
            require(file_digest(public_stage/dataset["pilot_calendar_file"]) == dataset["pilot_calendar_sha256"],
                    "pilot calendar changed while freezing")
            for block in dataset["blocks"]:
                block["calendar_file"] = copy_public(compare.artifact_path(draft_path, block["calendar_file"]),
                                                     "calendars/"+block["block_id"]+".json")
                require(file_digest(public_stage/block["calendar_file"]) == block["calendar_sha256"],
                        "final calendar changed while freezing")
        bound.update(status="frozen", final_execution_allowed=True,
            freeze={"created_utc": datetime.now(timezone.utc).isoformat(),
                    "draft_file_sha256": check["draft_file_sha256"], "freezer_source_sha256": file_digest(__file__),
                    "no_final_model_execution_performed": True, "no_final_selection_or_rate_tuning": True})
        bound.pop("draft_reason", None)
        bound["protocol_sha256"] = digest({k: v for k, v in bound.items() if k != "protocol_sha256"})
        staged_protocol = public_stage/output.name
        staged_protocol.write_text(json.dumps(bound, indent=2, sort_keys=True)+"\n")
        compare.load_admitted_protocol(staged_protocol)
        load_source_metadata(bound, staged_protocol)
        (public_stage/"preparation-record.json").write_text(json.dumps({"status": "passed", "blocks": evidence,
            "private_files": 25, "final_requests": 320, "pilot_warmup_requests": 40,
            "calendar_count": 20, "raw_text_or_token_arrays_published": False,
            "tokenization": "Verified against prior admitted metadata; no new retokenization or final model execution"}, indent=2)+"\n")
        private_stage.rename(private_output)
        public_stage.rename(output.parent)
        return {"status": "frozen", "execute": True, "output": str(output),
                "protocol_file_sha256": file_digest(output), "protocol_sha256": bound["protocol_sha256"],
                "private_output": str(private_output), "final_requests": 320, "warmup_requests": 40,
                "measured_executions": 960, "server_launches": 60, "model_execution_performed": False}
    except BaseException:
        # Only this invocation's uniquely named staging directories are removed.
        if public_stage.exists(): shutil.rmtree(public_stage)
        if private_stage.exists(): shutil.rmtree(private_stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("check")
    check.add_argument("--draft", type=Path, required=True)
    bind = sub.add_parser("freeze")
    for name in ("draft", "config", "source-manifest", "pilot-admission", "suite-private", "private-output", "output"):
        bind.add_argument("--"+name, type=Path, required=True)
    bind.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = check_draft(args.draft)[2] if args.action == "check" else freeze(
        args.draft, args.config, args.source_manifest, args.pilot_admission,
        args.suite_private, args.private_output, args.output, execute=args.execute)
    summary = {k: v for k, v in result.items() if k != "calendars"}
    if "calendars" in result:
        summary["calendars_verified"] = len(result["calendars"])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
