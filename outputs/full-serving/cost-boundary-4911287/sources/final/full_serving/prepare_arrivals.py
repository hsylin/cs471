"""Select unused public Azure arrival blocks, without examining performance.

Selection uses only pinned source day row bounds and previously used row IDs.
Every source byte is subsequently read and hashed, with day bounds rechecked.
Timestamps remain unscaled; attaching independent WildChat content does not
reconstruct the original Azure requests or their input/output token lengths.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
import hashlib
import io
import json
from pathlib import Path

from generalization.external_arrivals import parse_timestamp


SEED = 471110
TARGETS = (
    ("conversation", "2024-05-13", "pilot", 24),
    ("code", "2024-05-10", "final", 64),
    ("code", "2024-05-11", "final", 64),
    ("code", "2024-05-12", "final", 64),
    ("conversation", "2024-05-16", "final", 64),
    ("conversation", "2024-05-17", "final", 64),
    ("conversation", "2024-05-18", "final", 64),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def merged_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if start < 0 or end < start:
            raise ValueError("Invalid previous source row interval")
        if result and start <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def overlaps(start, end, intervals):
    # Source indices and both endpoints are inclusive, including horizon rows.
    position = bisect_right(intervals, (end, float("inf"))) - 1
    return position >= 0 and intervals[position][1] >= start


def load_exclusions(results, output):
    intervals = {"code": [], "conversation": []}
    inventory = []
    for path in sorted(Path(results).rglob("arrival-manifest.json")):
        if path.resolve() == Path(output).resolve():
            continue
        document = json.loads(path.read_text())
        blocks = document.get("blocks")
        if not isinstance(blocks, list):
            raise ValueError(f"Unrecognized prior manifest: {path}")
        for block in blocks:
            trace = block["trace"]
            if trace not in intervals:
                raise ValueError(f"Unrecognized prior trace {trace}")
            start = block["row_start"]
            boundary = block.get("boundary_row", block["row_stop_exclusive"])
            intervals[trace].append((start, boundary))
        inventory.append({"path": str(path.relative_to(Path(results).parent)),
                          "sha256": file_sha256(path), "blocks": len(blocks)})
    return {trace: merged_intervals(rows) for trace, rows in intervals.items()}, inventory


def choose_block(trace, day, split, jobs, row_start, row_stop, exclusions, seed=SEED):
    winner = None
    total = excluded = 0
    for start in range(row_start, row_stop - jobs, jobs + 1):
        total += 1
        if overlaps(start, start + jobs, exclusions):
            excluded += 1
            continue
        score = digest([seed, trace, day, start])
        if winner is None or (score, start) < (winner[0], winner[1]):
            winner = (score, start)
    if winner is None:
        raise ValueError(f"No unused candidate block: {trace}/{day}")
    score, start = winner
    return {"id": f"full-azure-{trace}-{day}-{split}", "trace": trace,
            "source_date_utc": day, "calendar_date_cluster": day, "split": split,
            "jobs": jobs, "row_start": start, "row_stop_exclusive": start + jobs,
            "boundary_row": start + jobs, "hash_rank_score": score,
            "candidate_bins": total, "excluded_candidate_bins": excluded,
            "eligible_candidate_bins": total - excluded,
            "discarded_partial_bin_rows": (row_stop - row_start) % (jobs + 1)}


class HashingReader(io.RawIOBase):
    def __init__(self, path):
        self.file = open(path, "rb")
        self.hasher = hashlib.sha256()
        self.bytes_read = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.file.readinto(buffer)
        if count:
            self.hasher.update(memoryview(buffer)[:count])
            self.bytes_read += count
        return count

    def close(self):
        self.file.close()
        super().close()


def extract_blocks(path, expected_sha256, blocks, expected_days):
    wanted = {index for block in blocks
              for index in range(block["row_start"], block["boundary_row"] + 1)}
    rows, days = {}, {}
    previous_day = None
    cache = {}
    raw = HashingReader(path)
    count = 0
    with io.TextIOWrapper(io.BufferedReader(raw, buffer_size=1 << 20),
                          encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]:
            raise ValueError("Unexpected Azure CSV schema")
        for index, row in enumerate(reader):
            if len(row) != 3:
                raise ValueError(f"Malformed Azure row {index}")
            day = row[0][:10]
            if previous_day is not None and day < previous_day:
                raise ValueError("Source days are not monotone; no sorting allowed")
            previous_day = day
            if day not in days:
                days[day] = [index, index + 1]
            else:
                days[day][1] = index + 1
            if index in wanted:
                ns, parsed_day, precision = parse_timestamp(row[0], cache)
                if parsed_day != day:
                    raise ValueError("Timestamp date mismatch")
                context, generated = int(row[1]), int(row[2])
                if min(context, generated) < 0:
                    raise ValueError("Negative source token metadata")
                rows[index] = {"row_index": index, "raw_timestamp": row[0], "timestamp_ns": ns,
                               "context_tokens": context, "generated_tokens": generated,
                               "fractional_digits": precision}
            count = index + 1
    if raw.hasher.hexdigest() != expected_sha256:
        raise ValueError("Source SHA256 mismatch")
    if days != expected_days:
        raise ValueError("Source day bounds differ from selection inventory")
    if set(rows) != wanted:
        raise ValueError("Selected rows absent from source")
    completed = []
    for block in blocks:
        selected = [rows[i] for i in range(block["row_start"], block["boundary_row"] + 1)]
        times = [row["timestamp_ns"] for row in selected]
        if any(a > b for a, b in zip(times, times[1:])):
            raise ValueError("Hash-selected block has inverted timestamps; do not resample")
        horizon = times[-1] - times[0]
        if horizon <= 0 or times[-1] <= times[-2]:
            raise ValueError("Hash-selected block has zero horizon/terminal tie; do not resample")
        body, boundary = selected[:-1], selected[-1]
        completed.append({**block, "source_sha256": expected_sha256,
                          "source_origin_ns": times[0], "original_offsets_ns": [t - times[0] for t in times[:-1]],
                          "original_horizon_ns": horizon, "row_indices": [r["row_index"] for r in body],
                          "raw_timestamps": [r["raw_timestamp"] for r in body],
                          "timestamp_ns": times[:-1], "boundary_timestamp": boundary["raw_timestamp"],
                          "boundary_timestamp_ns": times[-1], "terminal_gap_ns": times[-1] - times[-2],
                          "context_tokens": [r["context_tokens"] for r in body],
                          "generated_tokens": [r["generated_tokens"] for r in body],
                          "boundary_context_tokens": boundary["context_tokens"],
                          "boundary_generated_tokens": boundary["generated_tokens"],
                          "source_rows_sha256": digest([[r["raw_timestamp"], r["context_tokens"], r["generated_tokens"]] for r in selected])})
    return completed, {"rows": count, "bytes": raw.bytes_read, "sha256": raw.hasher.hexdigest(),
                       "day_row_bounds": days}


def prepare(source_dir, download_manifest, results, output):
    results, output = Path(results), Path(output)
    download = {entry["name"]: entry for entry in json.loads(Path(download_manifest).read_text())}
    exclusions, prior = load_exclusions(results, output)
    source_records, selected, scans = {}, [], {}
    inventory_dir = results / "generalization/arrivals-v1"
    for trace, suffix in (("code", "code"), ("conversation", "conv")):
        filename = f"AzureLLMInferenceTrace_{suffix}_1week.csv"
        entry = download[filename]
        inventory_path = inventory_dir / f"{trace}-scan.json"
        inventory = json.loads(inventory_path.read_text())
        if inventory["source_sha256"] != entry["sha256"]:
            raise ValueError("Day inventory/source pin mismatch")
        bounds = {day["date_utc"]: [day["row_start"], day["row_stop_exclusive"]] for day in inventory["days"]}
        blocks = [choose_block(service, date, split, count, *bounds[date], exclusions[trace])
                  for service, date, split, count in TARGETS if service == trace]
        complete, scan = extract_blocks(Path(source_dir) / filename, entry["sha256"], blocks, bounds)
        selected.extend(complete)
        scans[trace] = scan
        source_records[trace] = {**entry, "day_inventory": str(inventory_path.relative_to(results.parent)),
                                 "day_inventory_sha256": file_sha256(inventory_path)}
        print(json.dumps({"trace": trace, "verified_rows": scan["rows"], "selected_blocks": len(complete)}), flush=True)
    order = {(trace, date): i for i, (trace, date, _, _) in enumerate(TARGETS)}
    selected.sort(key=lambda block: order[(block["trace"], block["source_date_utc"])])
    license_path = inventory_dir / "source-documentation/azure-license.txt"
    readme_path = inventory_dir / "source-documentation/azure-2024-readme.md"
    manifest = {"schema": "full_serving_arrivals_v1", "seed": SEED, "blocks": selected,
                "selection": {"procedure": "On fixed target dates, minimum SHA256 canonical JSON [seed,trace,date,row_start] among disjoint day-start-anchored jobs+1 row bins excluding every prior inclusive row range. Partial bins discarded. Validate selected timestamps without outcome-based replacement.",
                              "uses_performance_outcomes": False, "uses_burstiness_filter": False,
                              "target_dates_and_counts": [list(row) for row in TARGETS],
                              "source_day_inventory_verified_by_full_stream": True},
                "previous_manifests": prior, "excluded_inclusive_intervals": exclusions,
                "source_rows_and_boundaries_disjoint": True,
                "fresh_rows_not_new_calendar_days_relative_to_prior_studies": True,
                "final_calendar_date_clusters": sorted({b["source_date_utc"] for b in selected if b["split"] == "final"}),
                "sources": source_records, "full_source_scans": scans,
                "download_manifest_sha256": file_sha256(download_manifest),
                "license": {"name": "CC-BY-4.0", "attribution": "Microsoft Azure, AzurePublicDataset, LLM inference traces (2024)",
                            "url": "https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE",
                            "archived_license": str(license_path.relative_to(results.parent)), "sha256": file_sha256(license_path),
                            "source_readme": str(readme_path.relative_to(results.parent)), "source_readme_sha256": file_sha256(readme_path)},
                "mapping": "Each Azure invocation timestamp becomes one independent full-model WildChat request. Azure token counts are retained metadata, not imposed on generated requests. This combines independent arrival and content sources; it does not recover joint production requests.",
                "arrival_rescaling": "None in this source manifest. Common rates may be chosen using pilot only and recorded in separate frozen run calendars.",
                "selector_source_sha256": file_sha256(__file__)}
    manifest["content_sha256"] = digest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--download-manifest", required=True)
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/full-serving/arrivals-v1/arrival-manifest.json")
    args = parser.parse_args()
    manifest = prepare(args.source_dir, args.download_manifest, args.results, args.output)
    print(json.dumps({"output": args.output, "blocks": len(manifest["blocks"]),
                      "content_sha256": manifest["content_sha256"]}))


if __name__ == "__main__":
    main()
