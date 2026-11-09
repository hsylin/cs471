"""Resolve a fresh optimization template without inspecting demand or timings.

Pins the capture index, source corpus, implementation and optional pilot
selection. It writes an unfrozen file once; use optimization.analysis freeze
after review. --data-root additionally checks actual bank/tape bytes locally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from followup.payback import followup_fingerprint
from optimization.analysis import file_hash, validate_manifest
from optimization.common import source_fingerprint
from optimization.input_sources import validate_source_manifest
from skewbench.native import harness_fingerprint


def resolve(template, capture_index, sources, *, selection=None, data_root=None):
    manifest = json.loads(Path(template).read_text())
    if "freeze" in manifest:
        raise ValueError("Never modify a frozen manifest; start from the template")
    source_path, index_path = Path(sources), Path(capture_index)
    corpus = json.loads(source_path.read_text())
    source_audit = validate_source_manifest(corpus, source_path.parent)
    captures = json.loads(index_path.read_text())
    if not isinstance(captures, list):
        raise ValueError("Expected the capture script's list-valued capture-index.json")
    by_block = {}
    for row in captures:
        if row["block_id"] in by_block:
            raise ValueError("Duplicate capture block")
        by_block[row["block_id"]] = row
    wanted = manifest["matrix"]["block_ids"]
    source_blocks = {b["block_id"]: b for b in corpus["blocks"]}
    manifest["blocks"], manifest["banks"], manifest["tapes"] = [], [], []
    checked = []
    for block in wanted:
        source = source_blocks[block]
        captured = by_block[block]
        if source["split"] != manifest["split"] or captured["split"] != manifest["split"] or captured.get("correctness_pass") is not True:
            raise ValueError("Source/capture split or correctness differs")
        manifest["blocks"].append({"block_id":block,"source_document_ids":[s["source_id"] for s in source["sources"]]})
        bank_path = str(Path("banks") / Path(captured["bank_file"]).name)
        manifest["banks"].append({"block_id":block,"capture_path":bank_path,"capture_sha256":captured["capture_sha256"]})
        files = [(bank_path, captured["capture_sha256"])]
        for replay in captured.get("replays", []):
            name = Path(replay["file"]).name
            match = re.fullmatch(re.escape(block) + r"-(stable|transient|persistent|switching)-b(128|1024)\.npz", name)
            if match is None:
                raise ValueError(f"Unexpected capture-index replay filename: {name}")
            replay_path = str(Path("tapes") / name)
            manifest["tapes"].append({"block_id":block,"regime":match[1],"batch_tokens":int(match[2]),
                "capture_path":bank_path,"capture_sha256":captured["capture_sha256"],
                "replay_path":replay_path,"replay_sha256":replay["sha256"]})
            files.append((replay_path,replay["sha256"]))
        if data_root is not None:
            for relative, digest in files:
                if file_hash(Path(data_root)/relative) != digest:
                    raise ValueError(f"Actual data bytes differ: {relative}")
                checked.append(relative)
    legacy, _ = harness_fingerprint()
    followup, _ = followup_fingerprint()
    optimized, files = source_fingerprint()
    pins = {"legacy_harness_sha256":legacy,"followup_sha256":followup,"optimization_sha256":optimized}
    # The predecessor pins are a compatibility constraint, never silently reset.
    for name in ("legacy_harness_sha256","followup_sha256"):
        if manifest["environment"].get(name) != pins[name]:
            raise ValueError(f"Frozen predecessor changed: {name}")
    manifest["environment"].update(pins)
    manifest["required_record_metadata"].update(pins)
    if manifest["phase"] != "tuning":
        if selection is None:
            raise ValueError("Policy manifests require the completed pilot selection file")
        result = json.loads(Path(selection).read_text())
        if result.get("status") != "selected":
            raise ValueError("Pilot selection is not complete")
        manifest["selected_configs"] = result["selected_configs"]
        manifest["selection_sha256"] = file_hash(selection)
        manifest["tuning_manifest_sha256"] = result["config_hash"]
    manifest["source_manifest"] = {"path":"../optimization/corpus/sources.json","sha256":file_hash(source_path)}
    manifest["pinning"] = {"capture_index_sha256":file_hash(index_path),"source_audit":source_audit,
        "optimization_file_sha256":files,"data_files_locally_verified":checked,
        "data_hash_check_scope":"File bytes only; no demand statistics or performance inspected. Runtime rechecks actual bank/tape hashes.",
        "pinning_script_sha256":file_hash(__file__)}
    validate_manifest(manifest, frozen=False)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template",type=Path,required=True)
    p.add_argument("--capture-index",type=Path,required=True)
    p.add_argument("--sources",type=Path,default=ROOT/"optimization/corpus/sources.json")
    p.add_argument("--selection",type=Path)
    p.add_argument("--data-root",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    result = resolve(args.template,args.capture_index,args.sources,selection=args.selection,data_root=args.data_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write("\n")
    print(json.dumps({"phase":result["phase"],"blocks":len(result["blocks"]),"banks":len(result["banks"]),
                      "tapes":len(result["tapes"]),"optimization_sha256":result["environment"]["optimization_sha256"]}))


if __name__ == "__main__":main()
