"""Frozen kernel tuning and paired factorial analysis; no old-study mutation."""
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

GROUP_FIELDS = ("block_id", "regime", "batch_tokens", "repetition")
COMMON_HASHES = ("capture_sha256", "replay_sha256", "source_map_sha256", "source_weights_sha256")
CONTRASTS = {
    "kernel_B3": (("TUNED", "B3"), ("DEFAULT", "B3")),
    "kernel_SKIP": (("TUNED", "SKIP"), ("DEFAULT", "SKIP")),
    "policy_DEFAULT": (("DEFAULT", "B3"), ("DEFAULT", "SKIP")),
    "policy_TUNED": (("TUNED", "B3"), ("TUNED", "SKIP")),
    "combined_vs_default_SKIP": (("TUNED", "B3"), ("DEFAULT", "SKIP")),
}
INFERENTIAL = {"kernel_B3", "kernel_SKIP"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def manifest_hash(manifest):
    return canonical_hash({k: v for k, v in manifest.items() if k != "freeze"})


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_manifest(manifest, *, frozen=True):
    if manifest.get("optimization_schema_version") != 1 or manifest.get("phase") not in {"tuning", "validation", "final"}:
        raise ValueError("unsupported optimization manifest")
    if manifest.get("split") != ("final" if manifest["phase"] == "final" else "pilot"):
        raise ValueError("phase/split mismatch")
    matrix = manifest["matrix"]
    fields = ["block_ids", "batch_tokens"] + (["candidate_ids", "domains"] if manifest["phase"] == "tuning" else ["kernel_variants", "policies", "regimes"])
    for field in fields:
        values = matrix[field]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"matrix.{field} must be nonempty and unique")
    if type(matrix["repetitions"]) is not int or matrix["repetitions"] < 1:
        raise ValueError("positive integer repetition count required")
    documents, blocks = set(), set()
    for block in manifest["blocks"]:
        source_ids = block["source_document_ids"]
        if block["block_id"] in blocks or not source_ids or len(source_ids) != len(set(source_ids)) or documents.intersection(source_ids):
            raise ValueError("blocks must contain unique disjoint source documents")
        blocks.add(block["block_id"]); documents.update(source_ids)
    if blocks != set(matrix["block_ids"]):
        raise ValueError("block provenance inventory differs")
    if documents.intersection(manifest.get("excluded_source_document_ids", [])):
        raise ValueError("new source documents overlap prior studies")
    if manifest["phase"] == "tuning":
        if set(matrix["candidate_ids"]) != set(manifest["tuning"]["candidate_configs"]) or "DEFAULT" not in matrix["candidate_ids"]:
            raise ValueError("candidate table differs from matrix")
        bank_keys = [b["block_id"] for b in manifest["banks"]]
        if len(bank_keys) != len(blocks) or set(bank_keys) != blocks or any(not _sha(b.get("capture_sha256")) for b in manifest["banks"]):
            raise ValueError("complete pinned bank inventory required")
    else:
        if set(matrix["kernel_variants"]) != {"DEFAULT", "TUNED"} or set(matrix["policies"]) != {"B3", "SKIP"}:
            raise ValueError("the fixed two-by-two factorial design is required")
        tapes = manifest["tapes"]
        tape_keys = [(t["block_id"], t["regime"], t["batch_tokens"]) for t in tapes]
        expected = set(itertools.product(matrix["block_ids"], matrix["regimes"], matrix["batch_tokens"]))
        if len(tape_keys) != len(expected) or set(tape_keys) != expected:
            raise ValueError("complete tape inventory required")
        if any(not _sha(t.get(f)) for t in tapes for f in ("capture_sha256", "replay_sha256")):
            raise ValueError("invalid tape hash")
        for batch in matrix["batch_tokens"]:
            selected = manifest["selected_configs"][str(batch)]
            configs = manifest["tuning"]["candidate_configs"]
            if selected["candidate_id"] not in configs or selected["config"] != configs[selected["candidate_id"]]:
                raise ValueError("selected config is outside the frozen candidate grid")
    plan = manifest["analysis"]
    if plan["bootstrap_samples"] < 100 or plan["minimum_source_blocks"] < 2:
        raise ValueError("invalid bootstrap design")
    digest = manifest_hash(manifest)
    if frozen and manifest.get("freeze", {}).get("sha256") != digest:
        raise ValueError("missing or changed freeze")
    return digest


def freeze_manifest(source, destination, *, selection=None, pilots=()):
    manifest = json.loads(Path(source).read_text())
    if "freeze" in manifest:
        raise ValueError("source is already frozen")
    if manifest["phase"] != "tuning":
        if selection is None:
            raise ValueError("validation/final requires a frozen pilot selection output")
        result = json.loads(Path(selection).read_text())
        if result.get("status") != "selected" or manifest["selected_configs"] != result["selected_configs"]:
            raise ValueError("manifest selection differs from completed pilot selection")
        manifest["selection_sha256"] = file_hash(selection)
        manifest["tuning_manifest_sha256"] = result["config_hash"]
    if manifest["phase"] == "final":
        if not pilots:
            raise ValueError("final requires frozen tuning and validation manifest references")
        docs = {d for b in manifest["blocks"] for d in b["source_document_ids"]}
        phases, hashes = set(), []
        for path in pilots:
            prior = json.loads(Path(path).read_text()); hashes.append(validate_manifest(prior)); phases.add(prior["phase"])
            if prior["split"] != "pilot" or docs.intersection(d for b in prior["blocks"] for d in b["source_document_ids"]):
                raise ValueError("pilot/final document overlap or wrong split")
            if prior["phase"] == "tuning" and hashes[-1] != manifest["tuning_manifest_sha256"]:
                raise ValueError("referenced tuning manifest differs from the selection source")
            if prior["phase"] == "validation" and (
                    prior.get("selection_sha256") != manifest["selection_sha256"]
                    or prior.get("tuning_manifest_sha256") != manifest["tuning_manifest_sha256"]
                    or prior.get("selected_configs") != manifest["selected_configs"]):
                raise ValueError("validation and final must use the same frozen pilot selection")
        if not {"tuning", "validation"}.issubset(phases):
            raise ValueError("both frozen pilot phases required")
        manifest["pilot_manifest_sha256"] = hashes
    validate_manifest(manifest, frozen=False)
    def unresolved(value):
        if isinstance(value, dict): return any(unresolved(v) for v in value.values())
        if isinstance(value, list): return any(unresolved(v) for v in value)
        return isinstance(value, str) and value.startswith("UNSET")
    if unresolved(manifest):
        raise ValueError("unresolved manifest fields")
    manifest["freeze"] = {"algorithm": "sha256", "sha256": manifest_hash(manifest)}
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False); handle.write("\n")
    destination.chmod(0o444)
    return manifest


def load_records(paths):
    rows = []
    for path in map(Path, paths):
        with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line); row["_source"] = f"{path}:{number}"; rows.append(row)
    return rows


def _record_issues(row, manifest, *, tuning=False):
    issues = []
    expected = {"optimization_schema_version": 1, "split": manifest["split"], "phase": manifest["phase"],
                "measurement_kind": "gpu_kernel_tuning" if tuning else "gpu_optimization",
                "status": "complete", "correctness_pass": True, "evidence_eligible": True,
                "steps_completed": manifest["tuning"]["forwards_per_trial"] if tuning else manifest["trace"]["measured_steps"],
                **manifest.get("required_record_metadata", {})}
    if not tuning:
        expected.update(observation_warmup_steps=manifest["trace"]["warmup_steps"], raw_trace_steps=manifest["trace"]["raw_steps"])
    for field, value in expected.items():
        if row.get(field) != value: issues.append(f"{field}_mismatch")
    seconds = row.get("elapsed_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
        issues.append("invalid_elapsed_seconds")
    return issues


def _config_issues(row, config_id, config, default_config=None):
    errors = []
    for field, value in (("kernel_config_id", config_id), ("kernel_override_config", config), ("kernel_config_sha256", canonical_hash(config))):
        if field not in row or row[field] != value: errors.append(f"{field}_mismatch")
    if not isinstance(row.get("resolved_kernel_config"), dict): errors.append("missing_resolved_kernel_config")
    elif config is not None and row["resolved_kernel_config"] != config:
        errors.append("resolved_kernel_config_differs_from_override")
    elif config is None and default_config is not None and row["resolved_kernel_config"] != default_config:
        errors.append("resolved_kernel_config_differs_from_frozen_default")
    return errors


def _matched_issues(rows, document_ids):
    ref = rows[0]; errors = []
    for field in COMMON_HASHES:
        if not _sha(ref.get(field)) or any(row.get(field) != ref.get(field) for row in rows): errors.append(f"{field}_missing_or_mismatched")
    for field in ("allocation_id", "hostname", "process_run_id"):
        if not ref.get(field) or any(row.get(field) != ref[field] for row in rows): errors.append(f"{field}_missing_or_mismatched")
    if any(row.get("source_prompt_ids") != document_ids for row in rows): errors.append("source_prompt_ids_mismatch")
    return errors


def select_configs(manifest, records):
    digest = validate_manifest(manifest)
    if manifest["phase"] != "tuning": raise ValueError("selection is pilot-tuning only")
    m, tuning = manifest["matrix"], manifest["tuning"]
    keys = list(itertools.product(m["block_ids"], m["domains"], m["batch_tokens"], range(m["repetitions"]), m["candidate_ids"]))
    fields = ("block_id", "domain", "batch_tokens", "repetition", "candidate_id")
    expected, index = set(keys), {}
    for row in records:
        key = tuple(row.get(f) for f in fields)
        if row.get("config_hash") != digest or key not in expected or key in index: raise ValueError("tuning hash/key/duplicate error")
        index[key] = row
    missing = [dict(zip(fields, k)) for k in keys if k not in index]
    if missing: return {"status": "incomplete", "config_hash": digest, "missing": missing, "selected_configs": {}}
    docs = {b["block_id"]: b["source_document_ids"] for b in manifest["blocks"]}
    banks = {b["block_id"]: b["capture_sha256"] for b in manifest["banks"]}
    errors, ratios = {}, {}
    for block, domain, batch, repetition in itertools.product(m["block_ids"], m["domains"], m["batch_tokens"], range(m["repetitions"])):
        group = (block, domain, batch, repetition); default = index[(*group, "DEFAULT")]
        for candidate in m["candidate_ids"]:
            row = index[(*group, candidate)]
            invalid = _record_issues(row, manifest, tuning=True) + _config_issues(row, candidate, tuning["candidate_configs"][candidate], manifest.get("default_configs",{}).get(str(batch)))
            if not invalid:
                invalid += _matched_issues([default, row], docs[block])
                if row["capture_sha256"] != banks[block]: invalid.append("capture_manifest_mismatch")
            errors.setdefault((batch, candidate), []).extend({"group": group, "issue": issue} for issue in invalid)
            if not invalid and not _record_issues(default, manifest, tuning=True):
                ratios[(*group, candidate)] = math.log(row["elapsed_seconds"]) - math.log(default["elapsed_seconds"])
    selected, reports = {}, []
    for batch in m["batch_tokens"]:
        if errors[(batch, "DEFAULT")]:
            return {"status": "invalid_default", "config_hash": digest, "selected_configs": {}, "errors": errors[(batch,"DEFAULT")]}
        options = []
        for candidate in m["candidate_ids"]:
            issue = errors[batch, candidate]
            cells = []
            if not issue:
                for block, domain in itertools.product(m["block_ids"], m["domains"]):
                    value = statistics.mean(ratios[block,domain,batch,rep,candidate] for rep in range(m["repetitions"]))
                    cells.append({"block_id":block,"domain":domain,"ratio":math.exp(value),"log_ratio":value})
            aggregate = math.exp(statistics.mean(c["log_ratio"] for c in cells)) if cells else None
            eligible = bool(cells) and all(c["ratio"] <= tuning["max_pilot_cell_ratio"] for c in cells)
            entry = {"batch_tokens":batch,"candidate_id":candidate,"ratio":aggregate,"eligible":eligible,"cells":cells,"issues":issue}
            reports.append(entry)
            if eligible: options.append(entry)
        fastest = min(options, key=lambda row: (row["ratio"], m["candidate_ids"].index(row["candidate_id"])))
        choice = "DEFAULT" if 1. <= fastest["ratio"] * tuning["tie_ratio"] else fastest["candidate_id"]
        selected[str(batch)] = {"candidate_id":choice,"config":tuning["candidate_configs"][choice]}
    return {"optimization_schema_version":1,"status":"selected","config_hash":digest,"expected_trials":len(keys),
            "selected_configs":selected,"candidate_results":reports,
            "rule":"Per batch, minimum equal-cell geometric paired runtime among complete correct candidates with every cell≤1.02; default wins within1%; frozen candidate order breaks ties.",
            "scope":"Pilot selection only; no final-data inference or new routing algorithm."}


def _quantile(values, p):
    values = sorted(values); pos=(len(values)-1)*p; low=int(pos); high=min(low+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(pos-low)


def block_estimate(logs, plan, family=1):
    values=[logs[k] for k in sorted(logs)]
    result={"source_blocks":len(values),"ratio":None,"ci95":None,"adjusted_ci":None,"family_size":family,"status":"missing"}
    if not values:return result
    result.update(ratio=math.exp(statistics.mean(values)),status="insufficient_blocks")
    if len(values)<plan["minimum_source_blocks"]:return result
    rng=random.Random(plan["bootstrap_seed"])
    boot=[math.exp(statistics.mean(rng.choices(values,k=len(values)))) for _ in range(plan["bootstrap_samples"])]
    result.update(ci95=[_quantile(boot,.025),_quantile(boot,.975)],
                  adjusted_ci=[_quantile(boot,.025/family),_quantile(boot,1-.025/family)],status="estimated")
    return result


def criterion(interval, limit):
    if interval is None:return "inconclusive"
    if interval[1]<=limit:return "pass"
    if interval[0]>limit:return "fail"
    return "inconclusive"


def analyze(manifest, records):
    digest=validate_manifest(manifest)
    if manifest["phase"]=="tuning":raise ValueError("use select for tuning records")
    m,plan=manifest["matrix"],manifest["analysis"]
    groups=list(itertools.product(m["block_ids"],m["regimes"],m["batch_tokens"],range(m["repetitions"])))
    arms=list(itertools.product(m["kernel_variants"],m["policies"]))
    expected={(g,a) for g in groups for a in arms};index={}
    for row in records:
        group=tuple(row.get(f) for f in GROUP_FIELDS);arm=(row.get("kernel_variant"),row.get("policy"));key=(group,arm)
        if row.get("config_hash")!=digest or key not in expected or key in index:raise ValueError("policy hash/key/duplicate error")
        index[key]=row
    docs={b["block_id"]:b["source_document_ids"] for b in manifest["blocks"]}
    tapes={(t["block_id"],t["regime"],t["batch_tokens"]):t for t in manifest["tapes"]}
    missing,invalid,pairs=[],[],[]
    for group in groups:
        absent=[list(a) for a in arms if (group,a) not in index]
        if absent:missing.append({**dict(zip(GROUP_FIELDS,group)),"arms":absent});continue
        rows={a:index[group,a] for a in arms};errors=_matched_issues(list(rows.values()),docs[group[0]])
        for (kernel,policy),row in rows.items():
            cfg={"candidate_id":"DEFAULT","config":None} if kernel=="DEFAULT" else manifest["selected_configs"][str(group[2])]
            errors += [f"{kernel}/{policy}:{e}" for e in _record_issues(row,manifest)+_config_issues(row,cfg["candidate_id"],cfg["config"],manifest.get("default_configs",{}).get(str(group[2])))]
            for field in ("capture_sha256","replay_sha256"):
                if row.get(field)!=tapes[group[:3]][field]:errors.append(f"{kernel}/{policy}:{field}_manifest_mismatch")
        for policy in m["policies"]:
            trajectory=rows["DEFAULT",policy].get("decision_trajectory_sha256")
            if not _sha(trajectory) or trajectory!=rows["TUNED",policy].get("decision_trajectory_sha256"):
                errors.append(f"{policy}:kernel_changed_decision_trajectory")
        if errors:invalid.append({**dict(zip(GROUP_FIELDS,group)),"issues":errors});continue
        pair=dict(zip(GROUP_FIELDS,group))
        for (kernel,policy),row in rows.items():pair[f"{kernel}_{policy}_seconds"]=row["elapsed_seconds"]
        for name,(a,b) in CONTRASTS.items():
            pair[name+"_log_ratio"]=math.log(rows[a]["elapsed_seconds"])-math.log(rows[b]["elapsed_seconds"])
        pair["interaction_log_ratio"]=pair["kernel_B3_log_ratio"]-pair["kernel_SKIP_log_ratio"]
        pairs.append(pair)
    complete=not missing and not invalid
    contrasts=list(CONTRASTS)+["interaction"]
    cells,cell_blocks=[],{}
    for regime,batch,contrast in itertools.product(m["regimes"],m["batch_tokens"],contrasts):
        block_logs={}
        for block in m["block_ids"]:
            values=[p[contrast+"_log_ratio"] for p in pairs if (p["block_id"],p["regime"],p["batch_tokens"])==(block,regime,batch)]
            if len(values)==m["repetitions"]:block_logs[block]=statistics.mean(values)
        cell_blocks[regime,batch,contrast]=block_logs
        family=plan["cell_families"]["stable" if regime=="stable" else "changing"] if contrast in INFERENTIAL else 1
        result=block_estimate(block_logs,plan,family)
        limit=plan["stable_ratio_limit"] if regime=="stable" else plan["useful_gain_ratio"]
        result.update(regime=regime,batch_tokens=batch,contrast=contrast,limit=limit,
                      criterion=criterion(result["adjusted_ci"],limit) if complete and contrast in INFERENTIAL else "inconclusive" if contrast in INFERENTIAL else "descriptive",
                      confirmatory_eligible=complete and manifest["phase"]=="final" and contrast=="kernel_B3")
        cells.append(result)
    aggregates=[]
    for group,contrast in itertools.product(("changing","stable","all"),contrasts):
        regimes=[r for r in m["regimes"] if (r=="stable" if group=="stable" else r!="stable" if group=="changing" else True)]
        constituents=[cell_blocks[r,b,contrast] for r in regimes for b in m["batch_tokens"]]
        logs={block:statistics.mean(c[block] for c in constituents) for block in m["block_ids"] if all(block in c for c in constituents)}
        result=block_estimate(logs,plan)
        limit=plan["stable_ratio_limit"] if group=="stable" else plan["useful_gain_ratio"]
        inferential=contrast in INFERENTIAL and group!="all"
        result.update(group=group,contrast=contrast,limit=limit,
                      criterion=criterion(result["ci95"],limit) if complete and inferential else "inconclusive" if inferential else "descriptive",
                      confirmatory_eligible=complete and manifest["phase"]=="final" and contrast=="kernel_B3" and group!="all")
        aggregates.append(result)
    primary=next(a for a in aggregates if a["contrast"]=="kernel_B3" and a["group"]=="changing")
    guard=[c["criterion"] for c in cells if c["contrast"]=="kernel_B3" and c["regime"]=="stable"]
    guard.append(next(a["criterion"] for a in aggregates if a["contrast"]=="kernel_B3" and a["group"]=="stable"))
    stable_status="fail" if "fail" in guard else "pass" if all(s=="pass" for s in guard) else "inconclusive"
    outcome="fail" if "fail" in (primary["criterion"],stable_status) else "pass" if primary["criterion"]==stable_status=="pass" else "inconclusive"
    return {"optimization_schema_version":1,"config_hash":digest,"phase":manifest["phase"],"split":manifest["split"],
            "status":"complete" if complete else "incomplete","expected_records":len(expected),"eligible_matched_groups":len(pairs),
            "missing":missing,"invalid_groups":invalid,"pairs":pairs,"cells":cells,"aggregates":aggregates,
            "primary_useful_gain":primary["criterion"],"stable_guard":stable_status,"acceptance":outcome,
            "claim_eligible":complete and manifest["phase"]=="final",
            "interval_scope":plan["interval_scope"],"scope":"Kernel configuration only; measured complete policy replay excludes input materialization, JIT and warmup; not full serving."}


def write_report(summary, destination, *, plots=True):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    (destination/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    for field in ("pairs","cells","aggregates"):
        with (destination/(field+".csv")).open("w",newline="") as handle:
            if summary[field]:
                writer=csv.DictWriter(handle,fieldnames=list(summary[field][0]));writer.writeheader();writer.writerows(summary[field])
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        chosen=[c for c in summary["cells"] if c["contrast"] in INFERENTIAL]
        keys=sorted({(c["regime"],c["batch_tokens"]) for c in chosen})
        fig,ax=plt.subplots(figsize=(8,5))
        for contrast,offset,color in (("kernel_B3",-.12,"#377eb8"),("kernel_SKIP",.12,"#e41a1c")):
            first=True
            for row in chosen:
                if row["contrast"]!=contrast or row["ratio"] is None:continue
                y=keys.index((row["regime"],row["batch_tokens"]))+offset
                if row["adjusted_ci"]:ax.plot(row["adjusted_ci"],[y,y],color=color)
                ax.plot(row["ratio"],y,"o",color=color,label=contrast if first else None);first=False
        ax.axvline(1,color="black",linewidth=.8);ax.axvline(.95,color="gray",linestyle="--",linewidth=.8)
        ax.set_yticks(range(len(keys)),[f"{r}, B{b}" for r,b in keys]);ax.invert_yaxis()
        ax.set_xlabel("Tuned / default completed runtime ratio");ax.set_title(f"Frozen kernel comparison: {summary['status']}")
        ax.legend();fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(destination/f"kernel-ratios.{ext}",dpi=160)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest="command",required=True)
    freeze=commands.add_parser("freeze");freeze.add_argument("--source",type=Path,required=True);freeze.add_argument("--destination",type=Path,required=True)
    freeze.add_argument("--selection",type=Path);freeze.add_argument("--pilots",type=Path,nargs="*",default=[])
    select=commands.add_parser("select");select.add_argument("--manifest",type=Path,required=True);select.add_argument("--records",type=Path,nargs="+",required=True);select.add_argument("--output",type=Path,required=True)
    report=commands.add_parser("report");report.add_argument("--manifest",type=Path,required=True);report.add_argument("--records",type=Path,nargs="+",required=True);report.add_argument("--output-dir",type=Path,required=True);report.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    if args.command=="freeze":print(freeze_manifest(args.source,args.destination,selection=args.selection,pilots=args.pilots)["freeze"]["sha256"])
    elif args.command=="select":
        result=select_configs(json.loads(args.manifest.read_text()),load_records(args.records));result["record_files_sha256"]={str(p):file_hash(p) for p in args.records}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as handle:json.dump(result,handle,indent=2,allow_nan=False);handle.write("\n")
        print(json.dumps({"status":result["status"],"selected_configs":result["selected_configs"]}))
    else:
        result=analyze(json.loads(args.manifest.read_text()),load_records(args.records));write_report(result,args.output_dir,plots=not args.no_plots)
        print(json.dumps({"status":result["status"],"primary_useful_gain":result["primary_useful_gain"],"stable_guard":result["stable_guard"],"acceptance":result["acceptance"]}))


if __name__=="__main__":main()
