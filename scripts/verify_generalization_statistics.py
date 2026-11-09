#!/usr/bin/env python3
"""Independent raw-clock/statistical audit of the frozen generalization study.

This file deliberately imports no production experiment, metrics, analysis or
inference module. NumPy supplies only PCG64 draws and array/quantile arithmetic.
It is an independent implementation by the same project team, not an external
laboratory replication. It neither executes GPU work nor selects new analyses.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("gsm8k", "mbpp", "wildchat")
TYPES = ("code", "conversation")
RATES = ("calibrated", "common")
ARMS = ("native", "shortcut", "combined")
CONTRASTS = {
    "combined_over_shortcut": ("shortcut", "combined"),
    "combined_over_native": ("native", "combined"),
    "shortcut_over_native": ("native", "shortcut"),
}
METRICS = ("response_mean_ns", "response_p95_ns")
SAMPLES, SEED = 100_000, 471512


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, expected, label="result"):
    """Compare a documented subset, preserving exact categorical decisions."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), label + ": keys")
        for key, value in expected.items():
            compare(actual[key], value, label + "/" + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ": length")
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, label + "/" + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label + ": value")
    else:
        require(type(actual) is type(expected) and actual == expected, label + ": exact value/type")


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + ": nonnegative integer required")
    return value


def linear_quantile(values, q):
    """Manual linear order statistic, independent of the production metrics."""
    values = sorted(values)
    require(bool(values), "Empty quantile")
    location = (len(values) - 1) * q
    lower = math.floor(location)
    upper = math.ceil(location)
    return float(values[lower] + (values[upper] - values[lower]) * (location - lower))


def raw_response(row):
    n = integer(row["n_jobs"], "n_jobs", 1)
    require(n == 256, "Exactly 256 scheduled jobs required")
    epoch = integer(row["epoch_monotonic_ns"], "epoch")
    horizon = integer(row["horizon_ns"], "horizon", 1)
    offsets = row["arrival_offsets_ns"]
    require(len(offsets) == n, "Calendar missing jobs")
    offsets = [integer(x, "arrival offset") for x in offsets]
    require(offsets == sorted(offsets) and offsets[0] == 0 and offsets[-1] < horizon,
            "Arrival calendar order/horizon")
    arrays = []
    for field in ("coordinator_admit_ns", "coordinator_start_ns", "coordinator_finish_ns"):
        require(len(row[field]) == n, field + ": omitted jobs")
        arrays.append([integer(x, field) for x in row[field]])
    admits, starts, finishes = arrays
    arrivals = [epoch + offset for offset in offsets]
    for i, (a, e, s, f) in enumerate(zip(arrivals, admits, starts, finishes)):
        require(a <= e <= s < f, "Invalid arrival/admission/start/finish clock order")
        require(i == 0 or finishes[i - 1] <= e, "Overlapping/out-of-order FCFS job")
    end = integer(row["observation_end_ns"], "observation end")
    require(end >= max(epoch + horizon, finishes[-1]), "Incomplete exposure/drain")
    for field in ("per_rank_start_ns", "per_rank_gpu_complete_ns"):
        require(len(row[field]) == 2 and all(len(x) == n for x in row[field]),
                "Both ranks must account for all jobs")
    for rank in range(2):
        for i in range(n):
            rs = integer(row["per_rank_start_ns"][rank][i], "rank start")
            rf = integer(row["per_rank_gpu_complete_ns"][rank][i], "rank completion")
            require(starts[i] <= rs <= rf <= finishes[i], "Rank outside acknowledged service envelope")
    responses = [f - a for f, a in zip(finishes, arrivals)]
    out = dict(response_mean_ns=sum(responses) / n,
               response_p95_ns=linear_quantile(responses, .95))
    stored = row["queue_metrics"]
    compare(stored, dict(complete=True, scheduled_jobs=n, completed_jobs=n, unfinished_jobs=0),
            "stored complete-run accounting")
    compare(stored["vectors_ns"]["response"], responses, "stored response vector")
    compare(stored["summary_ns"]["response"],
            dict(mean=out["response_mean_ns"], p95=out["response_p95_ns"]), "stored response metrics")
    return out


def plan_identity(plan, repo_root=None):
    require(plan.get("generalization_schema_version") == 1, "Generalization schema")
    phase = plan.get("phase")
    require(phase in ("calibration", "pilot", "final"), "Unknown phase")
    pin = digest({k: v for k, v in plan.items() if k != "freeze"})
    require(plan.get("freeze", {}).get("sha256") == pin, "Frozen plan hash mismatch")
    require(plan.get("arms") == list(ARMS) and plan.get("batch_tokens") == 128, "Three-arm B128 design")
    groups = defaultdict(list)
    seen = set()
    for item in plan["groups"]:
        gid, dataset = item["id"], item["dataset"]
        require(gid not in seen and dataset in DATASETS, "Duplicate/unknown content group")
        seen.add(gid)
        groups[dataset].append(gid)
    require(set(groups) == set(DATASETS), "Three dataset strata required")
    groups = {d: sorted(groups[d]) for d in DATASETS}
    sizes = {len(g) for g in groups.values()}
    allowed = {2} if phase == "pilot" or plan.get("split") == "pilot" else {6, 8}
    require(len(sizes) == 1 and sizes <= allowed, "Unequal/incorrect content allocation")
    if phase == "final":
        require(plan["allocation_decision"]["groups_per_dataset"] == next(iter(sizes)),
                "Final content count differs from pilot allocation")
    source = plan.get("execution_source_files", {})
    require(source and digest(source) == plan.get("execution_source_sha256"), "Execution source inventory digest")
    if repo_root is not None:
        root = Path(repo_root).resolve()
        files = {**source, **plan.get("protocol_files", {}),
                 "generalization/analysis.py": plan["analysis_sha256"],
                 "generalization/inference.py": plan["inference_sha256"]}
        for name, expected in files.items():
            path = (root / name).resolve()
            require(path.is_relative_to(root) and path.is_file() and file_digest(path) == expected,
                    "Frozen source/protocol changed: " + name)
    cases = {}
    dates_by_bundle = {}
    for item in plan["cases"]:
        g = item["block_id"]
        block, cal = item["source_block"], item["calendar"]
        key = (g, block["trace"], cal["rate_kind"])
        require(g in seen and key not in cases and item["dataset"] in groups
                and g in groups[item["dataset"]], "Malformed/duplicate case")
        require(key[1] in TYPES and key[2] in RATES and sorted(item["arm_order"]) == sorted(ARMS),
                "Malformed condition/arm order")
        require(cal["source_date_utc"] == block["source_date_utc"], "Calendar/source date mismatch")
        bundle = key[:2]
        require(bundle not in dates_by_bundle or dates_by_bundle[bundle] == block["source_date_utc"],
                "Date differs across rates")
        dates_by_bundle[bundle] = block["source_date_utc"]
        cases[key] = item
    if phase == "calibration":
        require(not cases, "Calibration contains treatment cases")
    else:
        require(set(cases) == set(itertools.product(seen, TYPES, RATES)), "Incomplete four-condition bundles")
        orders = Counter(tuple(c["arm_order"]) for c in cases.values())
        require(orders == Counter({p: len(cases) // 6 for p in itertools.permutations(ARMS)}),
                "Six arm permutations are not exactly balanced")
        for rate in RATES:
            subset = [c for key, c in cases.items() if key[2] == rate]
            require(Counter(tuple(c["arm_order"]) for c in subset)
                    == Counter({p: len(subset) // 6 for p in itertools.permutations(ARMS)}),
                    "Six arm permutations are not exactly balanced within rate " + rate)
        if phase == "final":
            actual_dates = {v for v in dates_by_bundle.values()}
            require(actual_dates == {f"2024-05-{i:02d}" for i in range(12, 19)}, "Seven joint actual final dates required")
    return pin, groups, cases


def recover_pairs(plan, records, groups, cases, pin):
    dataset_for = {g: d for d in DATASETS for g in groups[d]}
    phase = plan["phase"]
    expected = ({(g, None, "saturated", "native") for g in dataset_for} if phase == "calibration"
                else {(*case, arm) for case in cases for arm in ARMS})
    index, metrics, orders = {}, {}, defaultdict(list)
    for row in records:
        key = tuple(row[k] for k in ("content_group_id", "arrival_type", "rate_kind", "arm"))
        require(key in expected and key not in index, "Duplicate/unexpected trial: " + str(key))
        gid = key[0]
        compare(row, dict(status="complete", phase=phase, config_hash=pin,
                          execution_source_sha256=plan["execution_source_sha256"],
                          dataset=dataset_for[gid], block_id=gid, repetition=0,
                          batch_tokens=128, correctness_pass=True,
                          normal_completion_verified=True, resident_state_verified=True), "trial identity")
        if phase == "calibration":
            require(row["arrival_date"] is None and row["arrival_offsets_ns"] == [0] * 256
                    and row["horizon_ns"] == 1 and row["arm_order"] == ["native"], "Saturated calibration contract")
        else:
            case = cases[key[:-1]]
            compare(row, dict(arrival_offsets_ns=case["calendar"]["arrival_offsets_ns"],
                              horizon_ns=case["calendar"]["horizon_ns"], arrival_schedule=case["calendar"],
                              arrival_date=case["source_block"]["source_date_utc"],
                              arrival_block_id=case["source_block"]["id"], arm_order=case["arm_order"]),
                    "Frozen calendar identity")
            require(str(row["allocation_id"]) == str(plan["calibration"][gid]["allocation_id"]),
                    "Treatment/calibration allocation mismatch")
        require(str(row["allocation_id"]) == str(plan["hardware"]["allocation_id"]), "Plan allocation mismatch")
        index[key] = row
        metrics[key] = raw_response(row)
        orders[key[:-1]].append(row["arm"])
    require(set(index) == expected, "Missing trials (no omissions/exclusions allowed)")
    pairs = {}
    for key, case in sorted(cases.items()):
        require(orders[key] == case["arm_order"], "Observed arm order differs from frozen case")
        for contrast, (control, treatment) in CONTRASTS.items():
            a, b = metrics[(*key, control)], metrics[(*key, treatment)]
            pairs[(*key, contrast)] = dict(
                content_group_id=key[0], dataset=dataset_for[key[0]], arrival_type=key[1], rate_kind=key[2],
                arrival_date=case["source_block"]["source_date_utc"], contrast=contrast,
                control_arm=control, treatment_arm=treatment, control=a, treatment=b,
                ratios={m: b[m] / a[m] for m in METRICS},
                differences={m: b[m] - a[m] for m in METRICS},
                log_ratios={m: math.log(b[m]) - math.log(a[m]) for m in METRICS})
    return index, metrics, pairs


def bundle_values(pairs, groups, contrast, rate, metric, exclude_date=None):
    out = {}
    for gid in groups:
        values = [pairs[gid, typ, rate, contrast]["log_ratios"][metric] for typ in TYPES
                  if pairs[gid, typ, rate, contrast]["arrival_date"] != exclude_date]
        if values:
            out[gid] = math.fsum(values) / len(values)
    return out


def pilot_decision(pairs, groups):
    logs = {g: math.fsum(pairs[g, t, r, "combined_over_shortcut"]["log_ratios"]["response_mean_ns"]
                        for t, r in itertools.product(TYPES, RATES)) / 4
            for d in DATASETS for g in groups[d]}
    require(len(logs) == 6, "Pilot variance requires six content bundles")
    values = list(logs.values())
    center = math.fsum(values) / 6
    sd = math.sqrt(math.fsum((x - center) ** 2 for x in values) / 5)
    widths = {str(n): 2.571 * sd / math.sqrt(3 * n) for n in (6, 8)}
    n = 6 if widths["6"] <= .03 else 8
    dataset_sd = {d: abs(logs[groups[d][0]] - logs[groups[d][1]]) / math.sqrt(2) for d in DATASETS}
    return dict(groups_per_dataset=n, final_treatment_trials=3*n*2*2*3, pilot_content_groups=6,
                pilot_bundle_log_ratios=logs, pooled_sample_sd=sd, dataset_sample_sd=dataset_sd,
                projected_log_halfwidth=widths, uses_mean_benefit_for_selection=False)


def final_inference(pairs, groups):
    n = len(groups[DATASETS[0]])
    rng = np.random.default_rng(SEED)
    # Keep each dataset's call separate: this is the exact preregistered stream.
    draws = {d: rng.integers(0, n, size=(SAMPLES, n)) for d in DATASETS}
    draw_bytes = b"".join(np.asarray(draws[d], dtype="<i8").tobytes(order="C") for d in DATASETS)
    draw_digest = hashlib.sha256(draw_bytes).hexdigest()

    def endpoint(name, contrast, rate, metric, datasets=DATASETS, guard=False):
        logs, points, resamples = {}, [], []
        for d in datasets:
            values = bundle_values(pairs, groups[d], contrast, rate, metric)
            logs.update(values)
            array = np.array([values[g] for g in groups[d]], dtype=np.float64)
            points.append(float(array.mean()))
            resamples.append(np.take(array, draws[d]).mean(axis=1))
        distribution = np.exp(np.stack(resamples).mean(axis=0))
        upper = float(np.quantile(distribution, 1 - .05/7 if guard else .975, method="linear"))
        limit = 1.02 if guard else .95
        passed = upper <= limit if guard else upper < limit
        result = dict(name=name, contrast=contrast, rate_kind=rate, metric=metric, datasets=list(datasets),
                      ratio=math.exp(math.fsum(points)/len(points)), content_group_log_ratios=logs,
                      upper_confidence_bound=upper, ratio_limit=limit, passes=passed,
                      criterion="upper <= limit" if guard else "upper < limit",
                      status="pass" if passed else "not_established")
        if guard:
            result.update(one_sided_confidence_level=1-.05/7, guard_family_size=7)
        else:
            result["ci95"] = [float(np.quantile(distribution, .025, method="linear")), upper]
        return result

    primary = endpoint("calibrated_mean_response", "combined_over_shortcut", "calibrated", "response_mean_ns")
    guards = [endpoint(f"{d}_{r}_mean_response", "combined_over_shortcut", r, "response_mean_ns", (d,), True)
              for d, r in itertools.product(DATASETS, RATES)]
    guards.append(endpoint("calibrated_p95_response", "combined_over_shortcut", "calibrated", "response_p95_ns", guard=True))
    dates = sorted({p["arrival_date"] for p in pairs.values()})
    lodo = []
    for date in dates:
        means, retained, missing = [], {}, []
        for d in DATASETS:
            values = bundle_values(pairs, groups[d], "combined_over_shortcut", "calibrated", "response_mean_ns", date)
            retained[d] = len(values)
            if values:
                means.append(math.fsum(values.values()) / len(values))
            else:
                missing.append(d)
        ratio = None if missing else math.exp(math.fsum(means) / 3)
        lodo.append(dict(excluded_date=date, ratio=ratio, retained_content_groups=retained,
                         removed_arrival_bundles=sum(pairs[g,t,"calibrated","combined_over_shortcut"]["arrival_date"] == date
                                                     for d in DATASETS for g in groups[d] for t in TYPES),
                         passes=ratio is not None and ratio <= 1.02, missing_datasets=missing))
    valid_ratios = [x["ratio"] for x in lodo if x["ratio"] is not None]
    robustness = dict(ratio_limit=1.02, date_count=len(dates), dates=dates, results=lodo,
                      passes=all(x["passes"] for x in lodo),
                      ratio_range=[min(valid_ratios),max(valid_ratios)] if valid_ratios else None)
    gate = primary["passes"] and all(g["passes"] for g in guards) and robustness["passes"]
    if gate:
        secondary = endpoint("calibrated_mean_response_baseline", "combined_over_native", "calibrated", "response_mean_ns")
        secondary.update(confirmatory=True, gate_passed=True)
    else:
        logs, means = {}, []
        for d in DATASETS:
            part = bundle_values(pairs, groups[d], "combined_over_native", "calibrated", "response_mean_ns")
            logs.update(part)
            means.append(math.fsum(part.values()) / len(part))
        secondary = dict(name="calibrated_mean_response_baseline", contrast="combined_over_native",
                         ratio=math.exp(math.fsum(means)/3), content_group_log_ratios=logs, ci95=None,
                         upper_confidence_bound=None, ratio_limit=.95, passes=None,
                         status="not_tested_gate_closed", confirmatory=False, gate_passed=False)
    cells = []
    for d, typ, rate, contrast in itertools.product(DATASETS, TYPES, RATES, CONTRASTS):
        cells.append(dict(dataset=d, arrival_type=typ, rate_kind=rate, contrast=contrast, content_groups=n,
                          ratios={m: math.exp(math.fsum(pairs[g,typ,rate,contrast]["log_ratios"][m]
                                                       for g in groups[d])/n) for m in METRICS}))
    out = dict(schema="generalization_inference_v2", primary=primary, guards=guards,
               calendar_date_robustness=robustness, secondary_baseline=secondary, descriptive_cells=cells,
               incremental_acceptance="pass" if gate else "not_established",
               acceptance="pass" if gate and secondary["passes"] else "not_established",
               content_groups=3*n, paired_comparisons=len(pairs), treatment_trials=3*n*12,
               bootstrap=dict(samples=SAMPLES, seed=SEED, rng="numpy.random.default_rng/PCG64",
                              strata=list(DATASETS), groups_per_stratum=n,
                              unit="entire content bundle with two types and two rates", quantile_method="linear"))
    return out, draw_digest


def audit(plan, records, summary, *, repo_root=None):
    pin, groups, cases = plan_identity(plan, repo_root)
    index, metrics, pairs = recover_pairs(plan, records, groups, cases, pin)
    phase = plan["phase"]
    compare(summary, dict(valid=True, phase=phase, expected_records=len(index), observed_records=len(records),
                          valid_records=len(index), matched_triplets=len(cases), matched_pairs=len(pairs),
                          plan_sha256=pin, execution_source_sha256=plan["execution_source_sha256"],
                          errors=[], missing_keys=[]), "summary completeness")
    reported_trials = {}
    for t in summary["trial_summaries"]:
        key = tuple(t[k] for k in ("content_group_id", "arrival_type", "rate_kind", "arm"))
        require(key not in reported_trials, "Summary duplicate trial")
        reported_trials[key] = t
    require(set(reported_trials) == set(index), "Summary trial keys differ")
    for key, value in metrics.items():
        compare(reported_trials[key]["metrics"], value, "raw response vs trial summary " + str(key))
        compare(reported_trials[key], dict(dataset=index[key]["dataset"], arrival_date=index[key]["arrival_date"]), "trial summary identity")
    reported_pairs = {}
    for p in summary["paired_cases"]:
        key = tuple(p[k] for k in ("content_group_id", "arrival_type", "rate_kind", "contrast"))
        require(key not in reported_pairs, "Summary duplicate pair")
        reported_pairs[key] = p
    require(set(reported_pairs) == set(pairs), "Summary contrast matrix differs")
    for key, pair in pairs.items():
        compare(reported_pairs[key], pair, "raw response vs paired summary " + str(key))
    compare(summary["pairs_per_contrast"], dict(Counter(k[-1] for k in pairs)), "contrast counts")
    out = dict(passed=True, audit_schema_version=1, phase=phase, plan_sha256=pin,
               complete_trials=len(index), scheduled_jobs_verified=len(index)*256, matched_triplets=len(cases),
               paired_comparisons=len(pairs), pairs_per_contrast=dict(Counter(k[-1] for k in pairs)),
               content_groups_by_dataset={d: len(groups[d]) for d in DATASETS},
               production_modules_imported=False, exclusions=0,
               scope="Independent same-team CPU reimplementation; conditional on selected external trace material and one allocation. No arrival-population, GPU-allocation or external-laboratory replication claim.")
    if phase == "pilot":
        decision = pilot_decision(pairs, groups)
        compare(summary["allocation_decision"], decision, "pilot variance allocation")
        require(summary["confirmatory"] is None, "Pilot must not report confirmatory intervals")
        out.update(allocation_decision=decision, confirmatory=None)
    elif phase == "final":
        infer, draw_digest = final_inference(pairs, groups)
        compare(summary["confirmatory"], infer, "final inference")
        require(summary["allocation_decision"] is None, "Final cannot reselect content count")
        reported_digest = summary["confirmatory"]["bootstrap"].get("draw_indices_sha256")
        if reported_digest is not None:
            require(reported_digest == draw_digest, "Exact bootstrap draw digest differs")
        out.update(confirmatory=infer, bootstrap_draw_indices_sha256=draw_digest,
                   bootstrap_draw_digest_encoding="Concatenated dataset-order C-contiguous little-endian int64 tables, each 100000 x n",
                   production_draw_digest_comparison="matched" if reported_digest is not None else "not available in frozen production summary")
    else:
        require(summary["confirmatory"] is None and summary["allocation_decision"] is None, "Calibration is descriptive only")
        calibration = {}
        for key, row in index.items():
            starts, finishes = row["coordinator_start_ns"], row["coordinator_finish_ns"]
            elapsed = finishes[-1] - starts[0]
            value = dict(calibration_jobs=256, calibration_elapsed_ns=elapsed,
                         native_capacity_jobs_per_second=256e9/elapsed, native_mean_cycle_ns=elapsed/256,
                         mean_calibration_service_ns=sum(f-s for s,f in zip(starts,finishes))/256)
            compare(row["calibration_capacity"], value, "raw calibration capacity")
            calibration[key[0]] = value
        compare(summary["calibration"], calibration, "calibration summary")
        out["calibration"] = calibration
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "records", "summary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raw = args.records.read_bytes()
    payload = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
    evidence = dict(plan_file_sha256=file_digest(args.plan), records_file_sha256=hashlib.sha256(raw).hexdigest(),
                    records_jsonl_payload_sha256=hashlib.sha256(payload).hexdigest(),
                    summary_file_sha256=file_digest(args.summary), verifier_sha256=file_digest(__file__),
                    numpy_version=np.__version__)
    try:
        plan = json.loads(args.plan.read_text())
        summary = json.loads(args.summary.read_text())
        records = [json.loads(line) for line in payload.splitlines() if line.strip()]
        for name in ("records_file_sha256", "records_jsonl_payload_sha256"):
            if name in summary:
                # Compression may change on archival; the uncompressed payload is authoritative.
                if name.endswith("payload_sha256"):
                    require(summary[name] == evidence[name], "Summary references a different raw payload")
        result = {**audit(plan, records, summary, repo_root=ROOT), **evidence}
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        result = dict(passed=False, error=str(exc), **evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "phase", "complete_trials", "paired_comparisons", "error") if k in result}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
