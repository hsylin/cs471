"""Bind exact generation targets to the previous natural-EOS OFF records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

LENGTH_FIELDS = ("output_lengths_file", "output_lengths_sha256", "output_reference_file", "output_reference_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference_targets(raw, item, cap):
    require(raw.endswith(b"\n"), "incomplete reference record")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    headers = [r for r in rows if r.get("type") == "run_header"]
    summaries = [r for r in rows if r.get("type") == "run_summary"]
    records = sorted((r for r in rows if r.get("type") == "request"), key=lambda r: r["request_index"])
    require(len(headers) == len(summaries) == 1 and len(rows) == len(records) + 2,
            "one complete reference episode required")
    header, summary = headers[0], summaries[0]
    require(header.get("arm") == "off" and header.get("ignore_eos") is False
            and header.get("max_tokens") == cap and header.get("requests_sha256") == item["requests_sha256"],
            "reference must be the unchanged natural-EOS OFF input and output budget")
    require([r["request_id"] for r in records] == item["request_ids"]
            and summary.get("valid") is True and summary.get("failed_requests") == 0
            and summary.get("completed_requests") == len(records) == item["request_count"],
            "reference membership or completion differs")
    if item.get("calendar_sha256"):
        require(header.get("calendar_sha256") == item["calendar_sha256"], "reference arrival calendar differs")
    targets = {}
    for r in records:
        count = r.get("usage", {}).get("completion_tokens")
        require(r.get("success") is True and r.get("finish_reason") in ("stop", "length")
                and type(count) is int and 0 < count <= cap, "invalid natural reference output length")
        targets[r["request_id"]] = count
    return targets


def validate_targets(protocol, protocol_path):
    if not protocol.get("equal_output_length", False):
        return {}
    root = Path(protocol_path).resolve().parent
    result = {}
    for dataset in protocol["datasets"]:
        cap = protocol["generation_by_dataset"][dataset["dataset"]]["max_tokens"]
        for item in [dataset["warmup"], *dataset["blocks"]]:
            raw = {}
            for stem in ("output_lengths", "output_reference"):
                name = Path(item[stem + "_file"])
                require(not name.is_absolute() and ".." not in name.parts, "output-length artifact path must be confined")
                data = (root / name).read_bytes()
                require(hashlib.sha256(data).hexdigest() == item[stem + "_sha256"], "output-length evidence bytes changed")
                raw[stem] = data
            targets = json.loads(raw["output_lengths"])
            require(targets == reference_targets(raw["output_reference"], item, cap),
                    "targets must equal previous OFF lengths, without outcome-dependent adjustment")
            result[item["output_lengths_file"]] = targets
    return result
