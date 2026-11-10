"""Read-only audit of reused ShareGPT content under BurstGPT arrival timing.

Run from the repository root. This is a supplementary diagnostic, never a
sixth main corpus. Existing output artifacts are not overwritten.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from full_serving.audit_pilots import audit_cohort, read_json, read_jsonl, require
from full_serving.data import digest, file_digest


def audit(directory, main_directory, suite_manifest, private_directory, source_block, source_csv):
    manifest = read_json(suite_manifest)
    require(manifest["manifest_sha256"] == digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}),
            "suite manifest seal differs")
    cohort = next(c for c in manifest["cohorts"] if c["dataset"] == "sharegpt")
    result = audit_cohort(directory, cohort, private_directory)
    require(result["status"] == "passed" and result["native_counter_reconciliation"] == "matched",
            "supplementary phases/config/counters are not complete and reconciled")
    splits = read_json(directory/"pilot-splits.json")
    require(splits == read_json(main_directory/"pilot-splits.json"), "supplementary inputs differ from main ShareGPT pilot")
    for stage in ("smoke", "calibration", "external"):
        for arm in ("off", "on"):
            _, rows = read_jsonl(directory/arm/(stage+".jsonl"))
            header = next(r for r in rows if r["type"] == "run_header")
            require(header["run_id"] == f"full-pilot-sharegpt-burstgpt-diagnostic-{arm}-{stage}"
                    and header["block_id"] == f"sharegpt-burstgpt-diagnostic-pilot-{stage}",
                    "supplementary request records carry a different replay identity")
    source = read_json(source_block)
    require(source.get("dataset") == "burstgpt" and source.get("kind") == "metadata_only_no_prompt_text"
            and source.get("count") == 24, "expected the bounded metadata-only BurstGPT diagnostic")
    require(file_digest(source_csv) == source["source_sha256"], "raw BurstGPT CSV hash differs")
    selected, total = [], 0
    start = source["source_start_row_index"]
    with source_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == source["source_columns"], "BurstGPT CSV columns differ")
        for index, row in enumerate(reader):
            if start <= index <= start+24:
                selected.append(row)
            total += 1
    require(total == source["source_total_rows"] and len(selected) == 25
            and start == int(hashlib.sha256(source["seed"].encode()).hexdigest(), 16) % (total-24),
            "BurstGPT fixed-seed contiguous source selection differs")
    require([r["source_row_index"] for r in source["rows"]] == list(range(start, start+24))
            and [r["raw"] for r in source["rows"]] == selected[:-1], "BurstGPT selected rows differ from original CSV")
    timestamps = [Decimal(r["Timestamp"])*1_000_000_000 for r in selected]
    require(all(t == t.to_integral_value() for t in timestamps), "non-integer source nanoseconds")
    offsets = [int(t-timestamps[0]) for t in timestamps[:-1]]
    next_offset = int(timestamps[-1]-timestamps[0])
    horizon = max(next_offset, offsets[-1]+1_000_000_000)
    require(offsets == sorted(offsets) and source["original_offsets_ns"] == offsets
            and source["original_horizon_ns"] == horizon and source["next_source_offset_ns"] == next_offset,
            "BurstGPT raw timestamp offsets or horizon differ")

    calendar = read_json(directory/"pilot-calendar.json")
    block = {**source, "split": "pilot"}
    require(calendar["source_block"] == block and calendar["request_ids"] == splits["external"]["ids"]
            and calendar["absolute_arrivals"] is True and calendar["source_request_sizes_executed"] is False,
            "executed external calendar provenance differs")
    calibration_path = directory/"off/calibration.jsonl"
    _, calibration = read_jsonl(calibration_path)
    summary = next(r for r in calibration if r["type"] == "run_summary")
    requests = [r for r in calibration if r["type"] == "request"]
    elapsed = (max(max(r["end_ns"] for r in requests), summary["epoch_monotonic_ns"]+summary["offered_horizon_ns"])
               - summary["epoch_monotonic_ns"])/1e9
    rate = .6*len(requests)/elapsed
    target = Fraction(24*10**9, 1)/Fraction(str(rate))
    scale = target/horizon
    require(calendar["offered_requests_per_s"] == rate
            and calendar["arrival_offsets_ns"] == [int(x*scale) for x in offsets]
            and calendar["horizon_ns"] == int(target)
            and calendar["scale_numerator"] == scale.numerator and calendar["scale_denominator"] == scale.denominator,
            "executed calendar differs from OFF calibration and exact rational scaling")
    require(calendar["rate_selection"] == {
        "rule": "0.6 * off pilot burst completed requests / completed elapsed",
        "calibration_file_sha256": file_digest(calibration_path), "not_a_final_capacity_estimate": True},
        "calendar rate-selection provenance differs")
    arrival_manifest = {"blocks": [block], "content": "Repeated ShareGPT pilot; not BurstGPT prompts or fresh payloads",
                        "original_request_lengths_executed": False}
    arrival_manifest_hash = digest((json.dumps(arrival_manifest, indent=2, sort_keys=True)+"\n").encode())
    controller_source = read_json(directory/"controller-source.json")
    require(controller_source["arrival_manifest_sha256"] == arrival_manifest_hash
            and controller_source["data_manifest_sha256"] == file_digest(suite_manifest),
            "controller admission hashes differ from source manifests")
    report = {"schema": "supplementary_sharegpt_burstgpt_integrity_audit_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "diagnostic_id": "sharegpt-burstgpt-diagnostic", "main_content_dataset": "sharegpt",
              "new_corpus": False, "new_payloads": False, "main_corpus_count_remains": 5,
              "scope": "Repeated admitted ShareGPT pilot inputs with a separately calibrated BurstGPT metadata-only arrival block. This is not an authentic joint content/arrival trace or a causal cross-calendar comparison.",
              "suite_manifest_sha256": manifest["manifest_sha256"], "audit_source_sha256": file_digest(__file__),
              "cohort_integrity": result,
              "content_reuse": {"all_64_requests_and_8_32_24_partitions_match_main_sharegpt": True,
                                "main_partition_sha256": file_digest(main_directory/"pilot-splits.json")},
              "external_calendar_provenance": {"status": "passed", "source_csv_sha256": source["source_sha256"],
                  "source_block_sha256": file_digest(source_block), "source_rows_inclusive": [start, start+23],
                  "source_timestamp_ties_preserved": True, "source_horizon_ns": horizon,
                  "executed_calendar_sha256": file_digest(directory/"pilot-calendar.json"),
                  "admitted_arrival_manifest_sha256": arrival_manifest_hash,
                  "offered_requests_per_s": rate, "executed_horizon_ns": int(target),
                  "scale_numerator": scale.numerator, "scale_denominator": scale.denominator,
                  "source_token_sizes_executed": False, "source_failures_injected": False},
              "inference_claims": False, "raw_text_or_token_arrays_published": False}
    report["report_sha256"] = digest(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "main-directory", "suite-manifest", "private-directory", "source-block", "source-csv", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Choose a fresh output path")
    report = audit(args.directory, args.main_directory, args.suite_manifest, args.private_directory, args.source_block, args.source_csv)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "diagnostic_id": report["diagnostic_id"],
                      "new_corpus": False, "new_payloads": False, "output": str(args.output),
                      "report_sha256": report["report_sha256"]}))
