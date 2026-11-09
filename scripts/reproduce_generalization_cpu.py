#!/usr/bin/env python3
"""Reproduce archived generalization analyses/audits into a fresh directory.

No GPUs or model checkpoint are used. Run with the repository's CPU analysis
environment. A bundle lists explicit inputs; relative paths resolve against
--repo-root. Alternatively, supply phase-specific path flags and optionally
--write-bundle to save a portable, checksummed input inventory for later use.
This is same-team CPU reproducibility, not external laboratory replication.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

STUDIES = ("pilot-calibration", "pilot", "final-calibration", "final", "diagnostic")
COMPRESSION_METADATA = {"records_file_sha256", "records_compressed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record_identity(path):
    raw = Path(path).read_bytes()
    payload = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
    return {"records_file_sha256": hashlib.sha256(raw).hexdigest(),
            "records_jsonl_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "records_compressed": raw.startswith(b"\x1f\x8b")}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resolve(repo, path):
    p = Path(path).expanduser()
    return (repo / p).resolve() if not p.is_absolute() else p.resolve()


def portable(repo, path):
    p = Path(path).resolve()
    return str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p)


def compare(actual, expected, path="summary", differences=None):
    """Strict identities and tolerant floating arithmetic; no inference imports."""
    differences = differences if differences is not None else []
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), path + ": key set differs")
        for key in expected:
            compare(actual[key], expected[key], path + "/" + key, differences)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path + ": length differs")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, path + "/" + str(i), differences)
    elif type(expected) in (float, int) and type(actual) in (float, int):
        require(math.isfinite(actual) and math.isfinite(expected), path + ": nonfinite number")
        if type(expected) is int and type(actual) is int:
            require(actual == expected, path + ": integer differs")
        else:
            require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), path + ": number differs")
            differences.append(abs(actual - expected))
    else:
        require(type(actual) is type(expected) and actual == expected, path + ": identity differs")
    return differences


def compare_summaries(regenerated, archived):
    # Only container-level compression metadata may differ. Payload SHA,
    # analysis SHA, identities, all endpoints and categorical gates remain checked.
    a = {k: v for k, v in regenerated.items() if k not in COMPRESSION_METADATA}
    b = {k: v for k, v in archived.items() if k not in COMPRESSION_METADATA}
    errors = compare(a, b)
    return dict(matched=True, ignored_metadata_keys=sorted(COMPRESSION_METADATA),
                floating_values_compared=len(errors), maximum_absolute_numeric_difference=max(errors, default=0.0))


def inputs_from_flags(args, repo):
    studies = []
    for name in STUDIES:
        prefix = name.replace("-", "_")
        fields = {key: getattr(args, prefix + "_" + key) for key in ("plan", "records", "summary", "capture_root")}
        if not any(fields.values()):
            continue
        if fields["capture_root"] is None and name in ("pilot-calibration", "diagnostic"):
            fields["capture_root"] = args.pilot_capture_root
        if fields["capture_root"] is None and name == "final-calibration":
            fields["capture_root"] = args.final_capture_root
        require(all(fields.values()), name + ": plan, records, summary and capture root required")
        item = {"name": name, **{("archived_summary" if k == "summary" else k): portable(repo, resolve(repo, v))
                                for k, v in fields.items()}}
        roots = getattr(args, prefix + "_inputs_root") or []
        item["inputs_roots"] = [portable(repo, resolve(repo, p)) for p in roots]
        if name == "pilot" and args.pilot_allocation:
            item["archived_allocation"] = portable(repo, resolve(repo, args.pilot_allocation))
        studies.append(item)
    require(studies, "Supply a bundle or explicit study inputs")
    return {"schema_version": 1, "studies": studies}


def authenticate_bundle(bundle, repo):
    require(bundle.get("schema_version") == 1, "Unsupported bundle schema")
    studies = bundle.get("studies", [])
    names = [x.get("name") for x in studies]
    require(len(names) == len(set(names)) and all(x in STUDIES for x in names), "Duplicate/unknown study names")
    require("pilot" in names, "The pilot allocation check is required")
    require("final-calibration" not in names or "final" in names, "Final calibration requires final treatment inputs")
    verified = []
    for name in STUDIES:
        if name not in names:
            continue
        original = next(x for x in studies if x["name"] == name)
        item = dict(original)
        for field in ("plan", "records", "archived_summary"):
            require(resolve(repo, item[field]).is_file(), name + ": missing " + field)
        capture = resolve(repo, item["capture_root"])
        require(capture.is_dir(), name + ": capture root missing")
        plan_path = resolve(repo, item["plan"])
        plan = json.loads(plan_path.read_text())
        frozen = canonical_hash({k: v for k, v in plan.items() if k != "freeze"})
        require(plan.get("freeze", {}).get("sha256") == frozen, name + ": frozen plan hash differs")
        expected_phase = "calibration" if name.endswith("calibration") else name
        require(plan.get("phase") == expected_phase, name + ": plan phase differs")
        if expected_phase == "calibration":
            require(plan["split"] == name.split("-")[0], name + ": calibration split differs")
        hashes = dict(plan_file_sha256=sha(plan_path), plan_content_sha256=frozen,
                      archived_summary_sha256=sha(resolve(repo, item["archived_summary"])),
                      **record_identity(resolve(repo, item["records"])))
        archived = json.loads(resolve(repo, item["archived_summary"]).read_text())
        require(archived.get("valid") is True, name + ": archived analysis invalid")
        require(archived.get("records_jsonl_payload_sha256") == hashes["records_jsonl_payload_sha256"],
                name + ": archived summary references another uncompressed payload")
        if "hashes" in original:
            # A bundle may accompany recompressed records; the payload must be
            # identical, while its .gz container hash is recorded anew.
            for key, value in original["hashes"].items():
                if key not in COMPRESSION_METADATA:
                    require(hashes[key] == value, name + ": pinned bundle hash differs: " + key)
        item["hashes"] = hashes
        item["inputs_roots"] = item.get("inputs_roots", [])
        if item.get("archived_allocation"):
            require(resolve(repo, item["archived_allocation"]).is_file(), "Archived allocation missing")
            allocation_hash = sha(resolve(repo, item["archived_allocation"]))
            if "archived_allocation_sha256" in original:
                require(original["archived_allocation_sha256"] == allocation_hash, "Pinned allocation archive hash differs")
            item["archived_allocation_sha256"] = allocation_hash
        verified.append(item)
    return {"schema_version": 1, "studies": verified,
            "scope": "Same-team CPU regeneration and independent reimplementation checks; not GPU rerun or external-laboratory replication."}


def verify_cross_study(studies, repo, generated):
    byname = {s["name"]: s for s in studies}
    checked = []
    for treatment in ("pilot", "final"):
        if treatment not in byname:
            continue
        name = treatment + "-calibration"
        if name not in byname:
            checked.append(dict(treatment=treatment, calibration_raw_reproduced=False,
                                note="Frozen calibration values are checked by execution audit; separate calibration raw inputs were not supplied."))
            continue
        t = json.loads(resolve(repo, byname[treatment]["plan"]).read_text())
        c = json.loads(resolve(repo, byname[name]["plan"]).read_text())
        require(t["calibration_plan_sha256"] == c["freeze"]["sha256"], name + ": parent plan pin differs")
        require(t["calibration_records_sha256"] in (byname[name]["hashes"]["records_jsonl_payload_sha256"],
                                                    byname[name]["hashes"]["records_file_sha256"]),
                name + ": parent records pin differs")
        checked.append(dict(treatment=treatment, calibration_raw_reproduced=True, plan_and_records_pins_match=True))
    if "final" in byname:
        final = json.loads(resolve(repo, byname["final"]["plan"]).read_text())
        allocation = json.loads((generated / "pilot/allocation.json").read_text())
        compare(final["allocation_decision"], allocation, "Frozen final allocation versus regenerated pilot decision")
        if "pilot-calibration" in byname:
            require(final["common_rate"]["pilot_calibration_sha256"]
                    in (byname["pilot-calibration"]["hashes"]["records_file_sha256"],
                        byname["pilot-calibration"]["hashes"]["records_jsonl_payload_sha256"]),
                    "Common rate does not reference supplied pilot calibration")
    return checked


def run_command(argv, repo, target, commands):
    target.parent.mkdir(parents=True, exist_ok=True)
    record = dict(argv=[str(x) for x in argv], working_directory=str(repo),
                  stdout=str(target.with_suffix(".stdout.txt")), stderr=str(target.with_suffix(".stderr.txt")))
    commands.append(record)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # These entry points are CPU-only; make an accidental GPU use fail closed.
    env["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    print(json.dumps({"running": [str(x) for x in argv[1:4]]}), flush=True)
    with Path(record["stdout"]).open("w") as out, Path(record["stderr"]).open("w") as err:
        process = subprocess.run([str(x) for x in argv], cwd=repo, env=env, stdout=out, stderr=err)
    record.update(returncode=process.returncode, elapsed_seconds=time.monotonic()-started)
    require(process.returncode == 0, "CPU command failed; retained logs: " + record["stderr"])


def reproduce(repo, output, bundle):
    require(not output.exists() and not output.is_symlink(), "Refusing existing output directory: " + str(output))
    require((repo / "generalization/analysis.py").is_file(), "Repository/archive root missing analysis source")
    verified = authenticate_bundle(bundle, repo)
    output.mkdir(parents=True)
    (output / "inputs.json").write_text(json.dumps(verified, indent=2) + "\n")
    manifest = dict(schema_version=1, passed=False, repository_root=str(repo), python_executable=sys.executable,
                    reproduction_script_sha256=sha(__file__), gpu_execution=False, commands=[], studies=[],
                    limitations=[verified["scope"],
                        "Captured banks and saved assignment artifacts are required; checkpoint/GPU execution is not rerun.",
                        "GPU arithmetic and weight correctness remain recorded attestations authenticated by the independent execution verifier.",
                        "Only supplied studies are reproduced. Archived failure attempts and unrelated historical studies are not rerun."])
    try:
        git_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=repo, capture_output=True, text=True)
        exact_checkout = git_root.returncode == 0 and Path(git_root.stdout.strip()).resolve() == repo
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True) if exact_checkout else None
        manifest["source_commit"] = git.stdout.strip() if git is not None and git.returncode == 0 else None
        manifest["source_identity_basis"] = "Frozen per-file hashes checked by the analysis and independent verifiers; git commit is checkout context only."
        for study in verified["studies"]:
            name = study["name"]
            directory = output / name
            directory.mkdir()
            plan, records, archive, capture = [resolve(repo, study[k]) for k in ("plan", "records", "archived_summary", "capture_root")]
            summary = directory / "summary.json"
            diagnostic = name == "diagnostic"
            module = "generalization_diagnostics.analysis" if diagnostic else "generalization.analysis"
            command = [sys.executable, "-m", module, "--plan", plan, "--records", records, "--output", summary]
            if name == "pilot":
                command += ["--allocation-output", directory / "allocation.json"]
            run_command(command, repo, directory / "analysis", manifest["commands"])
            reproduced = json.loads(summary.read_text())
            archived = json.loads(archive.read_text())
            comparison = compare_summaries(reproduced, archived)
            verifier = "verify_generalization_diagnostics.py" if diagnostic else "verify_generalization_records.py"
            execution_path = directory / "independent-execution-audit.json"
            command = [sys.executable, repo / "scripts" / verifier, "--plan", plan, "--records", records,
                       "--capture-root", capture, "--summary", summary, "--output", execution_path]
            if study["inputs_roots"]:
                command += ["--inputs-root", *[resolve(repo, p) for p in study["inputs_roots"]]]
            run_command(command, repo, directory / "execution-audit", manifest["commands"])
            require(json.loads(execution_path.read_text())["passed"] is True, name + ": execution audit failed")
            if not diagnostic:
                statistics_path = directory / "independent-statistics-audit.json"
                run_command([sys.executable, repo / "scripts/verify_generalization_statistics.py",
                             "--plan", plan, "--records", records, "--summary", summary, "--output", statistics_path],
                            repo, directory / "statistics-audit", manifest["commands"])
                require(json.loads(statistics_path.read_text())["passed"] is True, name + ": statistics audit failed")
            allocation_comparison = None
            if name == "pilot" and study.get("archived_allocation"):
                regenerated = json.loads((directory / "allocation.json").read_text())
                old = json.loads(resolve(repo, study["archived_allocation"]).read_text())
                numeric = compare(regenerated, old, "Pilot allocation archive")
                allocation_comparison = dict(matched=True, maximum_absolute_numeric_difference=max(numeric, default=0.0))
            manifest["studies"].append(dict(name=name, archived_summary_comparison=comparison,
                allocation_archive_comparison=allocation_comparison, independent_execution_passed=True,
                independent_statistics_passed=True if not diagnostic else None,
                statistics_scope="descriptive diagnostic, no confidence interval/acceptance" if diagnostic else "frozen phase-specific analysis",
                input_hashes=study["hashes"], output_files={p.name: sha(p) for p in directory.iterdir() if p.is_file()}))
        manifest["cross_study_provenance"] = verify_cross_study(verified["studies"], repo, output)
        manifest["passed"] = True
    except (ValueError, OSError, KeyError, TypeError) as exc:
        manifest["error"] = str(exc)
    (output / "reproduction-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--write-bundle", type=Path, help="Save authenticated inputs for reuse; refuses overwrite")
    parser.add_argument("--pilot-allocation", type=Path)
    for name in STUDIES:
        for field in ("plan", "records", "summary", "capture-root"):
            parser.add_argument("--" + name + "-" + field, type=Path)
        parser.add_argument("--" + name + "-inputs-root", type=Path, nargs="+")
    args = parser.parse_args()
    repo, output = args.repo_root.resolve(), args.output.resolve()
    if args.bundle:
        require(not any(getattr(args, name.replace("-", "_") + "_plan") for name in STUDIES),
                "Use either a bundle or explicit study flags")
        bundle = json.loads(resolve(repo, args.bundle).read_text())
    else:
        bundle = inputs_from_flags(args, repo)
    if args.write_bundle:
        path = args.write_bundle.resolve()
        require(not path.exists() and not path.is_symlink(), "Refusing existing bundle: " + str(path))
        verified = authenticate_bundle(bundle, repo)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(verified, indent=2) + "\n")
    manifest = reproduce(repo, output, bundle)
    print(json.dumps({k: manifest[k] for k in ("passed", "error") if k in manifest}
                     | {"studies": [s["name"] for s in manifest["studies"]], "output": str(output)}))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
