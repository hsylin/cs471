"""Close collected full-model results and rebuild both independent analyses.

Run only after the controller wrapper exits successfully. No GPU work, input
selection, retries, parameter tuning, or experimental source edits occur here.
The caller supplies two identical remote inventories taken around the final sync.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from full_serving import audit_final_statistics, custom_report, render_custom_report


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root, wrapper):
    root, wrapper = Path(root), Path(wrapper)
    paths = {"results/" + str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}
    paths.update({name: wrapper / name for name in ("controller.log", "controller.err", "completed.txt")})
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("symlinks are not accepted in collected raw evidence")
    return {name: {"bytes": p.stat().st_size, "sha256": digest(p)} for name, p in sorted(paths.items())}


def verify_source(source, repo):
    revision = source["declared_git_revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("controller must identify its exact Git commit")
    tracked = subprocess.run(["git", "ls-tree", "-r", "--name-only", revision, "--", "full_serving"],
                             cwd=repo, check=True, capture_output=True, text=True).stdout.splitlines()
    expected_files = {name for name in tracked if name.endswith(".py")
                      or (name.endswith(".sh") and len(Path(name).parts) == 2)}
    if set(source["files"]) != expected_files:
        raise ValueError("controller source inventory differs from its declared commit")
    for name, expected in source["files"].items():
        if not name.startswith("full_serving/") or ".." in Path(name).parts:
            raise ValueError("unexpected controller source path")
        contents = subprocess.run(["git", "show", revision + ":" + name], cwd=repo,
                                  check=True, capture_output=True).stdout
        if hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError("controller source differs from declared commit: " + name)
    if not source["files"]:
        raise ValueError("controller source manifest is empty")
    return revision


def verify_closure(root, wrapper, remote, repo):
    root, wrapper = Path(root), Path(wrapper)
    if (wrapper / "completed.txt").read_bytes() != b"controller_exit=0\n":
        raise ValueError("successful controller wrapper exit is required")
    state = json.loads((root / "status.json").read_bytes())
    schedule = json.loads((root / "schedule.json").read_bytes())
    if (state.get("state") != "comparison_completed" or len(schedule) != 60
            or state.get("completed_sessions") != [job["session_id"] for job in schedule]):
        raise ValueError("all sixty ordered sessions must have completed")
    if (remote.get("schema") != "cs471_stable_remote_inventory_v1"
            or remote.get("before") != remote.get("after") or not remote.get("after")):
        raise ValueError("two identical remote inventories around final collection are required")
    local = inventory(root, wrapper)
    if local != remote["after"]:
        raise ValueError("local public records differ from stable remote bytes")
    source = json.loads((root / "controller-source.json").read_bytes())
    revision = verify_source(source, repo)
    return {"schema": "cs471_full_resident_closure_v1", "status": "passed",
            "closed_utc": datetime.now(timezone.utc).isoformat(), "controller_exit": 0,
            "completed_sessions": 60, "source_revision": revision,
            "source_files_verified_against_git": len(source["files"]),
            "public_files_verified_remote_and_local": len(local), "files": local,
            "normal_eplb_commits_verified": False,
            "meaning": "Stable public raw bytes and declared source verified; not a speedup or numerical-equivalence claim."}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish each artifact atomically; the completion receipt is published last.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def finish(root, wrapper, remote, repo, output_prefix, audit_dir):
    root, repo, prefix, audit_dir = map(Path, (root, repo, output_prefix, audit_dir))
    if audit_dir.resolve().is_relative_to(root.resolve()) or prefix.resolve().is_relative_to(root.resolve()):
        raise ValueError("analysis outputs must be outside the immutable raw run")
    closure = verify_closure(root, wrapper, remote, repo)
    report = custom_report.build_report(root)
    if not report["complete"]:
        raise ValueError("raw closure passed but full runtime/request/report admission did not")
    report_path = prefix.with_suffix(".json")
    write(report_path, report)
    audit = audit_final_statistics.audit(root, report_path)
    write(audit_dir / "independent-statistics-audit.json", audit)
    if audit["status"] != "passed":
        raise ValueError("independent raw-record statistics audit did not pass")
    render_custom_report.main([str(report_path), "--output-prefix", str(prefix)])
    interpretation = {"schema": "cs471_full_resident_interpretation_v1", "report_sha256": digest(report_path),
        "scope": "Per-dataset exploratory full-model response latency; no pooled or confirmatory claim.",
        "datasets": [{"dataset": d["dataset"], "contrasts": {
            c: render_custom_report.comparison_notes(d, c) for c in ("custom/off", "custom/on")}}
            for d in report["datasets"]]}
    write(prefix.with_name(prefix.name + "-interpretation.json"), interpretation)
    write(audit_dir / "raw-closure.json", closure)
    artifacts = [report_path, prefix.with_suffix(".html"), prefix.with_suffix(".csv"),
                 prefix.with_name(prefix.name + "-interpretation.json"),
                 audit_dir / "independent-statistics-audit.json", audit_dir / "raw-closure.json"]
    receipt = {"schema": "cs471_full_resident_completion_v1", "status": "completed_and_audited",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "measured_sessions": 60,
        "measured_http_requests": 960, "raw_closure": "passed", "independent_statistics": "passed",
        "experiment_source_revision": closure["source_revision"],
        "analysis_source_sha256": {str(p.relative_to(repo)): digest(p) for p in (repo / "full_serving").glob("*.py")},
        "artifacts": {str(p): digest(p) for p in artifacts}, "automatic_superiority_claim": False}
    write(audit_dir / "completion.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "wrapper", "remote-inventory", "repo", "output-prefix", "audit-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = finish(args.run, args.wrapper, json.loads(args.remote_inventory.read_bytes()),
                    args.repo, args.output_prefix, args.audit_dir)
    print(json.dumps({k: result[k] for k in ("status", "measured_sessions", "measured_http_requests")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
