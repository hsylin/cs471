import copy
import json
from pathlib import Path

import pytest

from full_serving import verify_suite as check
from full_serving.data import digest, file_digest, normalized, MODEL_REVISION


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "return_dict": False, "truncation": False,
                          "add_generation_prompt": True, "enable_thinking": False}
        return list(range(7 + sum(len(message["content"].split()) for message in messages)))


def source_row(dataset, index):
    return {"id": f"{dataset}-{index}", "dataset": dataset,
            "source_id": f"source-{dataset}-{index}", "source_split": "train",
            "messages": [{"role": "user", "content": f"PRIVATE original {dataset} question {index} 文字\u2028intact"}]}


def selected_row(source, index):
    row = copy.deepcopy(source)
    messages = row["messages"]
    tokens = Tokenizer().apply_chat_template(messages, tokenize=True, return_dict=False, truncation=False,
                                             add_generation_prompt=True, enable_thinking=False)
    row.update(group_id=row["source_id"], input_tokens=len(tokens), max_tokens=2048,
               messages_sha256=digest(messages),
               normalized_messages_sha256=digest([{**message, "content": normalized(message["content"])} for message in messages]),
               first_user_normalized_sha256=digest(normalized(messages[0]["content"]).encode()),
               prompt_token_ids_sha256=digest(tokens), split="pilot" if index < 64 else "final",
               content_block="pilot-00" if index < 64 else f"final-{(index-64)//64:02d}")
    return row


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows))


def seal(path, manifest):
    manifest["manifest_sha256"] = digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    write_json(path, manifest)


def fixture_suite(tmp_path, monkeypatch):
    root, private = tmp_path / "repo", tmp_path / "private"
    wild_private, wild_manifest = tmp_path / "wild-private", tmp_path / "wild-manifest.json"
    wild = [selected_row(source_row("wildchat", index), index) for index in range(448)]
    write_rows(wild_private / "requests.jsonl", wild)
    write_json(wild_manifest, {"fixture": "parent admission independently tested by verify_data"})
    monkeypatch.setattr(check, "verify_bundle", lambda *args: {"manifest_file_sha256": file_digest(wild_manifest)})
    candidates, cohorts = {}, []
    for dataset in check.DATASETS:
        rows = wild[:128] if dataset == "wildchat" else [selected_row(source_row(dataset, i), i) for i in range(128)]
        cohort = {"dataset": dataset, "smoke_request_ids": [row["id"] for row in rows[:8]], "private_files": {}}
        for filename, subset in (("requests.jsonl", rows), ("pilot.jsonl", rows[:64]),
                                 ("final.jsonl", rows[64:]), ("smoke.jsonl", rows[:8])):
            path = private / dataset / filename
            write_rows(path, subset)
            cohort["private_files"][filename] = file_digest(path)
        if dataset == "wildchat":
            cohort.update(parent_manifest_sha256=file_digest(wild_manifest),
                          other_final_bundles="reserved, not discarded or measured")
        else:
            candidate_path = tmp_path / f"{dataset}-candidates.jsonl"
            write_rows(candidate_path, [source_row(dataset, i) for i in range(128)])
            candidates[dataset] = candidate_path
            candidate_sha = file_digest(candidate_path)
            source = {"dataset": dataset}
            if dataset in ("instructcoder", "numinamath"):
                source["private_candidates"] = {"sha256": candidate_sha}
            elif dataset == "longbench":
                source["candidate_payload_sha256"] = candidate_sha
            else:
                source["preparation_manifest"] = "sharegpt-preparation.json"
                write_json(root / "results/sharegpt-preparation.json", {"candidate_file_sha256": candidate_sha})
            source_path = root / "results" / f"{dataset}-source.json"
            write_json(source_path, source)
            cohort.update(candidate_file_sha256=candidate_sha,
                          source_manifest_path=str(source_path.relative_to(root)),
                          source_manifest_sha256=file_digest(source_path),
                          selected_count=128, pilot_count=64, final_candidate_count=64,
                          requests=[{key: value for key, value in row.items() if key in check.PUBLIC_FIELDS} for row in rows],
                          streaming_frame={"input_candidate_rows": 128})
        cohorts.append(cohort)
    prep = root / "full_serving/prepare_suite.py"
    prep.parent.mkdir(parents=True)
    prep.write_text("# fixture selector source\n")
    manifest = {"schema": "full_serving_separate_suite_v1", "source_model_revision": MODEL_REVISION,
                "selection_status": "pilot_ready_final_candidates_not_frozen", "datasets_kept_separate": True,
                "pooled_speedup_claim": False, "selection_order": list(check.DATASETS), "cohorts": cohorts,
                "formatting": {"truncation": False, "enable_thinking": False, "add_generation_prompt": True, "max_model_len": 32768},
                "generation": {"max_tokens": 2048, "temperature": 0, "ignore_eos": False},
                "wildchat_admission": {"manifest_file_sha256": file_digest(wild_manifest)},
                "preparation_sources": {str(prep.relative_to(root)): file_digest(prep)}}
    manifest_path = root / "results/suite-manifest.json"
    seal(manifest_path, manifest)
    return dict(private_dir=private, manifest_path=manifest_path, candidate_paths=candidates,
                wildchat_private=wild_private, wildchat_manifest=wild_manifest, repo_root=root), manifest


def test_full640_admission_checks_tokens_source_chain_and_does_not_publish_text(tmp_path, monkeypatch):
    inputs, _ = fixture_suite(tmp_path, monkeypatch)
    report = check.verify_suite(**inputs, tokenizer=Tokenizer())
    assert report["status"] == "passed" and report["selected_requests"] == 640
    assert report["suite_manifest_file_sha256"] == file_digest(inputs["manifest_path"])
    assert report["recomputed_tokenization"] is True and report["final_protocol_frozen"] is False
    assert report["additional_wildchat_finals_preserved"] == 320
    assert "PRIVATE original" not in json.dumps(report)
    share = report["cohorts"][1]["source_binding"]
    assert share["preparation_digest_previously_declared"] is False
    assert share["candidate_file_sha256"] == share["declared_candidate_sha256"]


def test_resealed_reordered_pilot_file_is_not_an_acceptable_subset(tmp_path, monkeypatch):
    inputs, manifest = fixture_suite(tmp_path, monkeypatch)
    path = inputs["private_dir"] / "instructcoder/pilot.jsonl"
    rows, _ = check.read_private(path)
    write_rows(path, rows[::-1])
    manifest["cohorts"][2]["private_files"]["pilot.jsonl"] = file_digest(path)
    seal(inputs["manifest_path"], manifest)
    with pytest.raises(ValueError, match="Ordered pilot/final subset"):
        check.verify_suite(**inputs)


@pytest.mark.parametrize("mutation", ["source_declaration", "candidate_bytes", "private_bytes", "selector_bytes", "seal"])
def test_fail_closed_on_changed_evidence(tmp_path, monkeypatch, mutation):
    inputs, manifest = fixture_suite(tmp_path, monkeypatch)
    if mutation == "source_declaration":
        path = inputs["repo_root"] / "results/sharegpt-preparation.json"
        write_json(path, {"candidate_file_sha256": "0" * 64})
    elif mutation == "candidate_bytes":
        path = inputs["candidate_paths"]["numinamath"]
        with path.open("ab") as handle:
            handle.write(b"\n")
    elif mutation == "private_bytes":
        path = inputs["private_dir"] / "longbench/requests.jsonl"
        with path.open("ab") as handle:
            handle.write(b"\n")
    elif mutation == "selector_bytes":
        (inputs["repo_root"] / "full_serving/prepare_suite.py").write_text("changed")
    else:
        manifest["seed"] = "tampered_without_reseal"
        write_json(inputs["manifest_path"], manifest)
    with pytest.raises(ValueError):
        check.verify_suite(**inputs)


def test_original_candidate_messages_cannot_change_even_when_file_digest_matches(tmp_path):
    source = source_row("instructcoder", 0)
    selected = selected_row(source, 0)
    source["messages"][0]["content"] = "Changed original prompt"
    path = tmp_path / "source.jsonl"
    write_rows(path, [source])
    with pytest.raises(ValueError, match="original messages changed"):
        check.verify_candidate_file(path, "instructcoder", file_digest(path), [selected])


def test_retokenization_and_context_checks_do_not_accept_stale_counts():
    row = selected_row(source_row("numinamath", 0), 0)
    row["input_tokens"] += 1
    with pytest.raises(ValueError, match="Recomputed"):
        check.validate_request(row, "numinamath", 32768, 2048, Tokenizer())
    row["input_tokens"] = 32768
    with pytest.raises(ValueError, match="context allowance"):
        check.validate_request(row, "numinamath", 32768, 2048)


def test_source_group_cannot_cross_splits_even_with_resealed_private_bytes(tmp_path, monkeypatch):
    inputs, manifest = fixture_suite(tmp_path, monkeypatch)
    cohort = manifest["cohorts"][4]
    path = inputs["private_dir"] / "longbench/requests.jsonl"
    rows, _ = check.read_private(path)
    rows[64]["group_id"] = rows[0]["group_id"]
    write_rows(path, rows)
    cohort["private_files"]["requests.jsonl"] = file_digest(path)
    seal(inputs["manifest_path"], manifest)
    with pytest.raises(ValueError, match="Source group crosses"):
        check.verify_suite(**inputs)
