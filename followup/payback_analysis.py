"""Frozen three-arm candidate diagnostics; source blocks, never repeats, are units.

Synthetic fixtures exercise this module but are not measurement evidence. This
module lives outside skewbench/ to preserve the original experiment fingerprint.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

ARMS = ("KEEP", "PREINSTALLED", "CHARGED")
CONTRASTS = {"headroom": ("PREINSTALLED", "KEEP"),
             "payback": ("CHARGED", "KEEP"),
             "transition": ("CHARGED", "PREINSTALLED")}
KEY_FIELDS = ("block_id", "regime", "batch_tokens", "horizon_steps", "candidate_id", "repetition")
SHARED_HASHES = ("source_map_sha256", "source_weights_sha256", "candidate_map_sha256",
                 "capture_sha256", "replay_sha256")


def manifest_hash(manifest):
    payload = {k: v for k, v in manifest.items() if k != "freeze"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_manifest(manifest, *, frozen=True):
    if manifest.get("payback_schema_version") != 1 or manifest.get("split") not in {"pilot", "final"}:
        raise ValueError("invalid payback schema or split")
    matrix = manifest["matrix"]
    for name in ("arms", "regimes", "batch_tokens", "horizons", "block_ids", "candidate_ids"):
        values = matrix[name]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"matrix.{name} must be nonempty and unique")
    if set(matrix["arms"]) != set(ARMS) or matrix["repetitions"] < 1:
        raise ValueError("all three arms and positive repetitions required")
    if manifest["analysis"]["primary_horizon"] not in matrix["horizons"]:
        raise ValueError("primary horizon absent")
    if manifest["analysis"]["bootstrap_samples"] < 100:
            raise ValueError("at least 100 bootstrap draws required")
    if manifest["analysis"]["minimum_independent_blocks"] < 2:
        raise ValueError("at least 2 independent blocks required for intervals")
    blocks = manifest.get("blocks", [])
    if {b["block_id"] for b in blocks} != set(matrix["block_ids"]) or len(blocks) != len(matrix["block_ids"]):
        raise ValueError("exact block provenance required")
    documents = set()
    for block in blocks:
        ids = block.get("source_document_ids", [])
        if not ids or len(set(ids)) != len(ids) or documents.intersection(ids):
            raise ValueError("source documents must be nonempty and disjoint across blocks")
        documents.update(ids)
    tapes = manifest.get("tapes", [])
    expected = set(itertools.product(matrix["block_ids"], matrix["regimes"], matrix["batch_tokens"]))
    actual = [(t["block_id"], t["regime"], t["batch_tokens"]) for t in tapes]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("exact tape inventory required")
    for tape in tapes:
        if not all(_sha(tape.get(k)) for k in ("capture_sha256", "replay_sha256")):
            raise ValueError("tape SHA pins required")
    if frozen and manifest.get("freeze", {}).get("sha256") != manifest_hash(manifest):
        raise ValueError("missing or changed frozen manifest")
    return manifest_hash(manifest)


def freeze_manifest(source, destination, pilots=()):
    manifest = json.loads(Path(source).read_text())
    if "freeze" in manifest:
        raise ValueError("source already frozen")
    validate_manifest(manifest, frozen=False)
    def unresolved(value):
        if isinstance(value, dict):
            return any(unresolved(v) for v in value.values())
        if isinstance(value, list):
            return any(unresolved(v) for v in value)
        return value is None or isinstance(value, str) and value.startswith("UNSET")
    if unresolved(manifest):
        raise ValueError("resolve all placeholders before freezing")
    if manifest["split"] == "final":
        if not pilots:
            raise ValueError("final requires frozen pilot references")
        docs = {d for b in manifest["blocks"] for d in b["source_document_ids"]}
        hashes = []
        for path in pilots:
            prior = json.loads(Path(path).read_text())
            hashes.append(validate_manifest(prior))
            if prior["split"] != "pilot" or docs & {d for b in prior["blocks"] for d in b["source_document_ids"]}:
                raise ValueError("pilot/final source documents overlap or wrong pilot split")
            if set(manifest["matrix"]["block_ids"]) & set(prior["matrix"]["block_ids"]):
                raise ValueError("pilot/final block IDs overlap")
        manifest["pilot_manifest_sha256"] = sorted(set(hashes))
    manifest["freeze"] = {"algorithm": "sha256", "sha256": manifest_hash(manifest)}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as out:
        json.dump(manifest, out, indent=2, allow_nan=False)
        out.write("\n")
    destination.chmod(0o444)
    return manifest


def load_records(paths):
    records = []
    for path in map(Path, paths):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    row["_source"] = f"{path}:{line_number}"
                    records.append(row)
    return records


def _expected(manifest, stage):
    m = manifest["matrix"]
    keys = []
    for key in itertools.product(m["block_ids"], m["regimes"], m["batch_tokens"],
                                 m["horizons"], m["candidate_ids"], range(m["repetitions"])):
        if manifest["split"] == "pilot" and key[-1] > 0:
            if stage == "screening" or key[1] == "stable" or key[3] != manifest["analysis"]["primary_horizon"]:
                continue
        keys.append(key)
    return keys


def _quantile(values, p):
    values = sorted(values)
    pos = p * (len(values) - 1)
    index = int(pos)
    return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (pos - index)


def estimate(block_logs, *, samples, seed, minimum_blocks, family_size=1):
    values = [block_logs[k] for k in sorted(block_logs)]
    result = {"independent_blocks": len(values), "ratio": None, "ci95": None,
              "family_adjusted_ci": None, "family_size": family_size, "status": "missing"}
    if not values:
        return result
    result.update(ratio=math.exp(statistics.mean(values)), status="insufficient_blocks")
    if len(values) < minimum_blocks:
        return result
    rng = random.Random(seed)
    draws = [math.exp(statistics.mean(rng.choices(values, k=len(values)))) for _ in range(samples)]
    alpha = .025 / family_size
    result.update(ci95=[_quantile(draws, .025), _quantile(draws, .975)],
                  family_adjusted_ci=[_quantile(draws, alpha), _quantile(draws, 1-alpha)], status="estimated")
    return result


def criterion(interval, limit):
    if interval is None:
        return "inconclusive"
    if interval[1] <= limit:
        return "pass"
    if interval[0] > limit:
        return "fail"
    return "inconclusive"


def analyze(manifest, records, *, stage="screening"):
    digest = validate_manifest(manifest)
    expected = _expected(manifest, stage)
    expected_set = set(expected)
    tapes = {(t["block_id"], t["regime"], t["batch_tokens"]): t for t in manifest["tapes"]}
    block_documents = {b["block_id"]: b["source_document_ids"] for b in manifest["blocks"]}
    index, issues, ignored = {}, [], []
    for row in records:
        if row.get("config_hash") != digest:
            raise ValueError("record config_hash mismatch; do not pool manifests")
        key = tuple(row.get(f) for f in KEY_FIELDS)
        arm = row.get("arm")
        if key not in expected_set:
            if manifest["split"] == "pilot" and stage == "screening" and row.get("repetition", 0) > 0:
                ignored.append(row.get("_source", repr(key)))
                continue
            raise ValueError(f"unexpected record key {key}")
        if arm not in ARMS or (key, arm) in index:
            raise ValueError(f"invalid or duplicate arm {key} {arm}")
        index[key, arm] = row
    pairs, missing, state_contracts, prefix_contracts = [], [], {}, {}
    for key in expected:
        rows = {arm: index.get((key, arm)) for arm in ARMS}
        absent = [arm for arm, row in rows.items() if row is None]
        if absent:
            missing.append({**dict(zip(KEY_FIELDS, key)), "arms": absent})
            continue
        errors = []
        reference = rows["KEEP"]
        tape = tapes[key[:3]]
        if reference.get("source_prompt_ids") != block_documents[key[0]]:
            errors.append("source_prompt_ids_manifest_mismatch")
        for arm, row in rows.items():
            elapsed = row.get("elapsed_seconds")
            if (row.get("payback_schema_version") != 1 or row.get("split") != manifest["split"]
                    or row.get("measurement_kind") != "gpu_payback" or row.get("status") != "complete"
                    or row.get("correctness_pass") is not True or row.get("evidence_eligible") is not True
                    or row.get("steps_completed") != key[3]
                    or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed) or elapsed <= 0):
                errors.append(f"{arm}:invalid_completed_measurement")
            for field in SHARED_HASHES:
                if not _sha(row.get(field)) or row.get(field) != reference.get(field):
                    errors.append(f"{arm}:{field}_missing_or_mismatched")
            for field in ("capture_sha256", "replay_sha256"):
                if row.get(field) != tape[field]:
                    errors.append(f"{arm}:{field}_manifest_mismatch")
            if not row.get("source_prompt_ids") or row.get("source_prompt_ids") != reference.get("source_prompt_ids"):
                errors.append(f"{arm}:source_prompt_ids_missing_or_mismatched")
            if not row.get("state_id") or row.get("state_id") != reference.get("state_id"):
                errors.append(f"{arm}:state_id_missing_or_mismatched")
            for field in ("allocation_id", "hostname", "process_run_id"):
                if not row.get(field) or row.get(field) != reference.get(field):
                    errors.append(f"{arm}:{field}_missing_or_mismatched")
            if arm != "CHARGED" and row.get("migration_bytes") != 0:
                errors.append(f"{arm}:timed_migration_bytes_nonzero")
            for field, value in manifest.get("required_record_metadata", {}).items():
                if row.get(field) != value:
                    errors.append(f"{arm}:{field}_manifest_mismatch")
        state_key = (key[0], key[1], key[2], key[4])
        state_contract = tuple(reference.get(field) for field in SHARED_HASHES)
        if state_key in state_contracts and state_contracts[state_key] != state_contract:
            errors.append("source_candidate_or_replay_changed_between_horizons_or_repetitions")
        prefix_contract = tuple(reference.get(field) for field in SHARED_HASHES if field != "candidate_map_sha256")
        if key[:3] in prefix_contracts and prefix_contracts[key[:3]] != prefix_contract:
            errors.append("source_prefix_or_replay_changed_between_candidate_families")
        if errors:
            issues.append({**dict(zip(KEY_FIELDS, key)), "issues": errors})
            continue
        state_contracts[state_key] = state_contract
        prefix_contracts[key[:3]] = prefix_contract
        out = dict(zip(KEY_FIELDS, key))
        out.update({f"{arm.lower()}_seconds": row["elapsed_seconds"] for arm, row in rows.items()})
        out["payback_seconds"] = rows["KEEP"]["elapsed_seconds"] - rows["CHARGED"]["elapsed_seconds"]
        out["headroom_seconds"] = rows["KEEP"]["elapsed_seconds"] - rows["PREINSTALLED"]["elapsed_seconds"]
        out["transition_seconds"] = rows["CHARGED"]["elapsed_seconds"] - rows["PREINSTALLED"]["elapsed_seconds"]
        for name, (numerator, denominator) in CONTRASTS.items():
            out[name + "_log_ratio"] = math.log(rows[numerator]["elapsed_seconds"]) - math.log(rows[denominator]["elapsed_seconds"])
        pairs.append(out)
    complete = not missing and not issues
    analysis, matrix = manifest["analysis"], manifest["matrix"]
    changing = [r for r in matrix["regimes"] if r != "stable"]
    nfamilies = len(matrix["candidate_ids"])
    cells = []
    by_cell = {}
    for candidate, horizon, regime, batch in itertools.product(matrix["candidate_ids"], matrix["horizons"], matrix["regimes"], matrix["batch_tokens"]):
        selected = [p for p in pairs if (p["candidate_id"], p["horizon_steps"], p["regime"], p["batch_tokens"]) == (candidate,horizon,regime,batch)]
        for contrast in CONTRASTS:
            logs = {}
            for block in matrix["block_ids"]:
                values = [p[contrast+"_log_ratio"] for p in selected if p["block_id"] == block]
                required = sum(k[0] == block and k[1:5] == (regime,batch,horizon,candidate) for k in expected)
                if len(values) == required and required:
                    logs[block] = statistics.mean(values)
            by_cell[candidate,horizon,regime,batch,contrast] = logs
            family = nfamilies * (len(changing) if regime != "stable" else 1) * len(matrix["batch_tokens"])
            result = estimate(logs, samples=analysis["bootstrap_samples"], seed=analysis["bootstrap_seed"],
                              minimum_blocks=analysis["minimum_independent_blocks"], family_size=family)
            limit = analysis["stable_ratio_limit"] if regime == "stable" else analysis["useful_gain_ratio"]
            result.update(candidate_id=candidate, horizon_steps=horizon, regime=regime, batch_tokens=batch,
                          contrast=contrast, limit=limit,
                          criterion=criterion(result["family_adjusted_ci"], limit) if contrast != "transition" else "descriptive")
            result["confirmatory_eligible"] = complete and manifest["split"] == "final" and horizon == analysis["primary_horizon"]
            if not complete and contrast != "transition":
                result["descriptive_criterion"] = result["criterion"]
                result["criterion"] = "inconclusive"
            cells.append(result)
    aggregates = []
    for candidate,horizon,group,contrast in itertools.product(matrix["candidate_ids"],matrix["horizons"],("changing","stable"),CONTRASTS):
        regimes = changing if group == "changing" else ["stable"]
        cell_logs = [by_cell[candidate,horizon,r,b,contrast] for r in regimes for b in matrix["batch_tokens"]]
        logs = {block:statistics.mean(c[block] for c in cell_logs) for block in matrix["block_ids"] if all(block in c for c in cell_logs)}
        result = estimate(logs, samples=analysis["bootstrap_samples"], seed=analysis["bootstrap_seed"],
                          minimum_blocks=analysis["minimum_independent_blocks"], family_size=nfamilies)
        limit = analysis["stable_ratio_limit"] if group == "stable" else analysis["useful_gain_ratio"]
        result.update(candidate_id=candidate,horizon_steps=horizon,group=group,contrast=contrast,limit=limit,
                      criterion=criterion(result["family_adjusted_ci"],limit) if contrast != "transition" else "descriptive")
        result["confirmatory_eligible"] = complete and manifest["split"] == "final" and horizon == analysis["primary_horizon"]
        if not complete and contrast != "transition":
            result["descriptive_criterion"] = result["criterion"]
            result["criterion"] = "inconclusive"
        aggregates.append(result)
    primary = analysis["primary_horizon"]
    verdicts = []
    for candidate in matrix["candidate_ids"]:
        relevant = [c for c in cells if c["candidate_id"] == candidate and c["horizon_steps"] == primary and c["regime"] != "stable"]
        head = [c for c in relevant if c["contrast"] == "headroom"]
        pay = [c for c in relevant if c["contrast"] == "payback"]
        if not complete:
            verdict = "incomplete"
        elif any(c["criterion"] == "pass" for c in pay):
            verdict = "charged_gain_without_resident_headroom_requires_timing_audit" if all(c["criterion"] == "fail" for c in head) else "useful_charged_payback_demonstrated"
        elif all(c["criterion"] == "fail" for c in head):
            verdict = "useful_resident_gain_ruled_out_in_tested_cells"
        elif any(c["criterion"] == "pass" for c in head) and all(c["criterion"] == "fail" for c in pay):
            verdict = "resident_gain_with_charged_gain_ruled_out"
        else:
            verdict = "inconclusive"
        stable = [c["criterion"] for c in cells if c["candidate_id"] == candidate and c["horizon_steps"] == primary and c["regime"] == "stable" and c["contrast"] == "payback"]
        stable += [a["criterion"] for a in aggregates if a["candidate_id"] == candidate and a["horizon_steps"] == primary and a["group"] == "stable" and a["contrast"] == "payback"]
        guard = "fail" if "fail" in stable else "pass" if all(s == "pass" for s in stable) else "inconclusive"
        primary_result = next(a["criterion"] for a in aggregates if a["candidate_id"] == candidate and a["horizon_steps"] == primary and a["group"] == "changing" and a["contrast"] == "payback")
        uniform = "fail" if "fail" in (guard, primary_result) else "pass" if guard == primary_result == "pass" else "inconclusive"
        verdicts.append({"candidate_id":candidate,"verdict":verdict,"stable_guard_status":guard,
                         "changing_aggregate_useful_gain_status":primary_result,"uniform_branch_criterion":uniform,
                         "permits_final_data_policy_selection":False})
    return {"payback_schema_version":1,"config_hash":digest,"split":manifest["split"],"stage":stage,
            "status":"complete" if complete else "incomplete","expected_arm_records":len(expected)*3,
            "eligible_pairs":len(pairs),"missing":missing,"invalid_pairs":issues,"ignored":ignored,
            "independent_unit":"source-document block, under the finite-corpus resampling assumption",
            "interval_interpretation":"Variation across fixed hand-selected books and CPython files under block resampling; disjoint documents do not establish population independence. Not production-workload population confidence intervals.",
            "pairs":pairs,"cells":cells,
            "aggregates":aggregates,"verdicts":verdicts,
            "primary_horizon":primary,
            "scope":"Conditional constructed teacher-forced-prefill replay; not natural online serving. Common planner formation cost excluded."}


def pilot_gate(summary, *, confirmation=False, minimum_seconds=.001):
    if summary["split"] != "pilot":
        raise ValueError("continuation may use pilot only")
    grouped = {}
    for pair in summary["pairs"]:
        if pair["regime"] != "stable" and pair["horizon_steps"] == 512:
            key = tuple(pair[k] for k in ("candidate_id","block_id","regime","batch_tokens"))
            grouped.setdefault(key,[]).append(pair["payback_seconds"])
    qualified=[]
    for key, values in grouped.items():
        if len(values) == (3 if confirmation else 1) and statistics.mean(values) >= minimum_seconds and sum(v > 0 for v in values) >= (2 if confirmation else 1):
            qualified.append({"candidate_id":key[0],"block_id":key[1],"regime":key[2],"batch_tokens":key[3],"mean_payback_seconds":statistics.mean(values),"positive_repetitions":sum(v>0 for v in values)})
    candidates=sorted({p["candidate_id"] for p in summary["pairs"]})
    decisions=[]
    for candidate in candidates:
        passed=summary["status"]=="complete" and len({s["block_id"] for s in qualified if s["candidate_id"]==candidate})>=2
        decision=("incomplete_measurements" if summary["status"]!="complete" else
                  "feasibility_demonstrated" if passed and confirmation else
                  "confirmation_required" if passed else
                  "robust_feasibility_not_demonstrated" if confirmation else "screen_did_not_qualify")
        decisions.append({"candidate_id":candidate,"continue":passed,"decision":decision})
    return {"pilot_only":True,"confirmation":confirmation,"minimum_payback_seconds":minimum_seconds,
            "qualified_states":qualified,"candidates":decisions}


def write_report(summary, destination, *, plots=True):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    for field in ("pairs","cells","aggregates"):
        rows=summary[field]
        with (destination/(field+".csv")).open("w",newline="") as handle:
            if rows:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        selected=[c for c in summary["cells"] if c["horizon_steps"]==summary["primary_horizon"] and c["contrast"] in {"headroom","payback"}]
        keys=sorted({(c["candidate_id"],c["regime"],c["batch_tokens"]) for c in selected})
        fig,ax=plt.subplots(figsize=(9,max(4,.4*len(keys))))
        for contrast,offset,color,label in [("headroom",-.12,"#377eb8","Preinstalled / keep"),("payback",.12,"#e41a1c","Charged / keep")]:
            first=True
            for c in selected:
                if c["contrast"]!=contrast or c["ratio"] is None:continue
                y=keys.index((c["candidate_id"],c["regime"],c["batch_tokens"]))+offset
                ci=c["family_adjusted_ci"]
                if ci:ax.plot(ci,[y,y],color=color)
                ax.plot(c["ratio"],y,"o",color=color,label=label if first else None);first=False
        ax.axvline(1,color="black",linewidth=.8)
        ax.axvline(.95,color="gray",linestyle="--",linewidth=.8,label="5% useful gain")
        ax.set_yticks(range(len(keys)),[f"{c} / {r} / B{b}" for c,r,b in keys])
        ax.invert_yaxis();ax.set_xlabel("Paired geometric mean runtime ratio (lower is faster)")
        ax.set_title(f"Same candidate, H={summary['primary_horizon']}; {summary['status']}\nSource-block intervals; common planner cost excluded")
        ax.legend(fontsize=8);fig.tight_layout()
        for extension in ("png","pdf"):fig.savefig(destination/f"payback-ratios.{extension}",dpi=160)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    freeze=sub.add_parser("freeze")
    freeze.add_argument("--source",type=Path,required=True);freeze.add_argument("--destination",type=Path,required=True)
    freeze.add_argument("--pilots",type=Path,nargs="*",default=[])
    report=sub.add_parser("report")
    report.add_argument("--manifest",type=Path,required=True);report.add_argument("--records",type=Path,nargs="+",required=True)
    report.add_argument("--output-dir",type=Path,required=True);report.add_argument("--pilot-stage",choices=["screening","confirmation"],default="screening")
    report.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    if args.command=="freeze":
        print(freeze_manifest(args.source,args.destination,args.pilots)["freeze"]["sha256"])
    else:
        result=analyze(json.loads(args.manifest.read_text()),load_records(args.records),stage=args.pilot_stage)
        if result["split"]=="pilot":result["pilot_gate"]=pilot_gate(result,confirmation=args.pilot_stage=="confirmation")
        write_report(result,args.output_dir,plots=not args.no_plots)
        print(json.dumps({"status":result["status"],"verdicts":result["verdicts"]}))


if __name__=="__main__":
    main()
