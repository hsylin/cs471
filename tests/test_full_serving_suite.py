"""Admission tests for separate full-model cohorts, without GPU dependencies."""
import copy
import json
from pathlib import Path
import stat
import sys
import types

import pytest

from full_serving import prepare_suite as suite
from full_serving.data import digest, file_digest


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        # Include chat-template overhead so the context guard is tested at its
        # actual boundary, not merely against raw user-text length.
        return list(range(7 + sum(len(m["content"].split()) for m in messages)))


def candidate(index, dataset="sharegpt", *, words=20, group=None, history=False):
    identity = f"{dataset}-{index}"
    messages = [{"role": "user", "content": " ".join([f"private-{identity}"] * words)}]
    if history:
        messages += [{"role": "assistant", "content": "Recorded assistant context.\n文字\u2028retained"},
                     {"role": "user", "content": f"Continue {identity}."}]
    return {"id": identity, "dataset": dataset, "source_id": f"source-{identity}",
            "source_split": "train", "group_id": group or f"group-{identity}",
            "messages": messages}


def hash_order(rows, dataset):
    return sorted(rows, key=lambda row: (digest([suite.SEED, dataset, row["id"]]), row["id"]))


def test_full_messages_survive_with_no_512_prefix_cap_and_original_history():
    original = [candidate(i, words=700 + i, history=True) for i in range(12)]
    untouched = copy.deepcopy(original)
    tokenizer = Tokenizer()
    selected, report = suite.select_cohort(original, "sharegpt", tokenizer, pilot=8, final=4)
    by_id = {row["id"]: row for row in original}
    assert original == untouched
    assert min(row["input_tokens"] for row in selected) > 512
    assert report["lengths"]["over_512"] == 12
    for row in selected:
        assert row["messages"] == by_id[row["id"]]["messages"]
        assert row["messages"][-1]["role"] == "user"
        assert row["messages_sha256"] == digest(row["messages"])
    assert all(options == {"tokenize": True, "return_dict": False, "truncation": False,
                           "add_generation_prompt": True, "enable_thinking": False}
               for _, options in tokenizer.calls)


def test_one_question_or_segment_per_group_before_pilot_final_split():
    rows = [candidate(f"{group}-{segment}", "longbench", group=f"document-{group}")
            for group in range(16) for segment in range(3)]
    selected, report = suite.select_cohort(rows, "longbench", Tokenizer(), pilot=8, final=8)
    pilot = [row for row in selected if row["split"] == "pilot"]
    final = [row for row in selected if row["split"] == "final"]
    assert len({row["group_id"] for row in selected}) == 16
    assert not {row["group_id"] for row in pilot} & {row["group_id"] for row in final}
    assert set(report["smoke_request_ids"]) == {row["id"] for row in pilot}
    assert report["rejection_counts"]["duplicate_source_group"] > 0
    reverse, _ = suite.select_cohort(list(reversed(rows)), "longbench", Tokenizer(), pilot=8, final=8)
    assert [row["id"] for row in reverse] == [row["id"] for row in selected]


def test_prior_normalized_full_messages_and_first_user_are_both_excluded():
    rows = [candidate(i, "instructcoder") for i in range(14)]
    ordered = hash_order(rows, "instructcoder")
    full_prior = [{"role": "user", "content": "already seen whole prompt"}]
    first_prior = [{"role": "user", "content": "same opening request"},
                   {"role": "assistant", "content": "old history"},
                   {"role": "user", "content": "old continuation"}]
    ordered[0]["messages"] = [{"role": "user", "content": "ＡＬＲＥＡＤＹ   SEEN whole prompt"}]
    ordered[1]["messages"] = [{"role": "user", "content": "SAME opening   request"},
                              {"role": "assistant", "content": "different history"},
                              {"role": "user", "content": "new continuation"}]
    selected, report = suite.select_cohort(rows, "instructcoder", Tokenizer(), pilot=8, final=4,
        seen_content={suite.message_key(full_prior), suite.message_key(first_prior)},
        seen_first={suite.first_user_key(full_prior), suite.first_user_key(first_prior)})
    assert not {ordered[0]["id"], ordered[1]["id"]} & {row["id"] for row in selected}
    assert report["rejection_counts"] == {"duplicate_messages": 1, "duplicate_first_user": 1}


def test_context_limit_charges_template_and_output_allowance_and_never_truncates():
    rows = [candidate(i, "numinamath", words=83) for i in range(13)]
    first = hash_order(rows, "numinamath")[0]
    first["messages"][0]["content"] += " extra"  # 84 + template7 + output10 >100
    tokenizer = Tokenizer()
    selected, report = suite.select_cohort(rows, "numinamath", tokenizer,
                                          pilot=8, final=4, max_context=100, max_output=10)
    assert first["id"] not in {row["id"] for row in selected}
    assert all(row["input_tokens"] == 90 and row["max_tokens"] == 10 for row in selected)
    assert report["rejected"] == [{"id": first["id"], "reason": "empty_or_over_context", "input_tokens": 91}]
    assert tokenizer.calls[0][0] == first["messages"]


def test_public_request_drops_raw_answers_and_nested_untrusted_metadata():
    row = {**candidate(1), "input_tokens": 700, "split": "pilot",
           "prompt": "PRIVATE PROMPT", "answer": "PRIVATE ANSWER", "context": "PRIVATE CONTEXT",
           "raw_metadata": {"original_messages": [{"content": "PRIVATE NESTED TEXT"}]}}
    public = suite.public_request(row)
    assert public["id"] == row["id"] and public["input_tokens"] == 700
    assert "PRIVATE" not in json.dumps(public)
    assert not {"messages", "prompt", "answer", "context", "raw_metadata"} & set(public)
    row["source_id"] = {"raw_prompt": "PRIVATE NESTED IN KNOWN FIELD"}
    with pytest.raises(ValueError, match="scalar"):
        suite.public_request(row)


def test_separate_source_arguments_required_and_burstgpt_is_not_a_text_dataset():
    full = [f"{name}=/private/{name}.jsonl" for name in suite.DATASETS]
    assert set(suite.mapping(full)) == set(suite.DATASETS)
    for invalid in (full[:-1], full + [full[0]], full + ["burstgpt=/private/trace.csv"]):
        with pytest.raises(ValueError):
            suite.mapping(invalid)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))


def test_streaming_hash_frame_keeps_global_order_and_rejects_late_duplicate(tmp_path):
    rows = [candidate(i) for i in range(80)]
    path = tmp_path / "candidates.jsonl"
    jsonl(path, rows)
    kept, report = suite.candidate_frame(path, "sharegpt", keep=12)
    expected = {row["id"] for row in hash_order(rows, "sharegpt")[:12]}
    assert {row["id"] for row in kept} == expected
    assert report["input_candidate_rows"] == 80
    assert report["retained_hash_ranks"] == 12
    jsonl(path, rows + [rows[-1]])
    with pytest.raises(ValueError, match="duplicate"):
        suite.candidate_frame(path, "sharegpt", keep=12)


def test_mixed_dataset_rejected_even_outside_retained_frame(tmp_path):
    path = tmp_path / "mixed.jsonl"
    jsonl(path, [candidate(i) for i in range(40)] + [candidate("wrong", "numinamath")])
    with pytest.raises(ValueError, match="Mixed"):
        suite.candidate_frame(path, "sharegpt", keep=12)


def test_suite_keeps_five_cohorts_and_all_448_wildchat_rows_in_exclusion_set(tmp_path, monkeypatch):
    """Exercise main's membership and public/private boundaries using CPU stubs.

    Source/tokenizer cryptographic admission is tested by their own modules;
    here the stubs isolate suite orchestration and prevent network/GPU work.
    """
    fake_repo = tmp_path / "repository"
    fake_package = fake_repo / "full_serving"
    fake_package.mkdir(parents=True)
    real_package = Path(suite.__file__).parent
    for name in ("prepare_suite.py", "data.py"):
        (fake_package / name).write_bytes((real_package / name).read_bytes())
    monkeypatch.setattr(suite, "__file__", str(fake_package / "prepare_suite.py"))
    (fake_repo / "results").mkdir()
    wild_private = tmp_path / "wildchat-private"
    wild = []
    for index in range(448):
        row = candidate(index, "wildchat")
        row.update(split="pilot" if index < 64 else "final",
                   content_block="pilot-00" if index < 64 else f"final-{(index-64)//64:02d}",
                   input_tokens=27, max_tokens=2048)
        wild.append(row)
    jsonl(wild_private / "requests.jsonl", wild)
    jsonl(wild_private / "pilot.jsonl", wild[:64])
    for block in range(6):
        jsonl(wild_private / f"final-{block:02d}.jsonl", wild[64 + block*64:128 + block*64])
    old_digests = {path.name: file_digest(path) for path in wild_private.glob("*.jsonl")}
    wild_manifest = tmp_path / "wildchat-manifest.json"
    wild_manifest.write_text(json.dumps({"smoke_request_ids": [row["id"] for row in wild[:8]]}))

    candidate_args, source_args = [], []
    banned_ids = set()
    previous_source_prompt = None
    original_messages = {}
    for dataset in suite.DATASETS:
        rows = [candidate(i, dataset) for i in range(140)]
        ranked = hash_order(rows, dataset)
        # These are from reserved final bundles which the suite does not use.
        ranked[0]["messages"] = copy.deepcopy(wild[447]["messages"])
        ranked[1]["messages"] = copy.deepcopy(wild[313]["messages"])
        banned_ids.update((ranked[0]["id"], ranked[1]["id"]))
        if previous_source_prompt is not None:
            ranked[2]["messages"] = copy.deepcopy(previous_source_prompt)
            banned_ids.add(ranked[2]["id"])
        previous_source_prompt = copy.deepcopy(ranked[3]["messages"])
        rows[0]["raw_metadata"] = {"answer": "DO_NOT_PUBLISH_ADAPTER_METADATA"}
        original_messages.update({row["id"]: copy.deepcopy(row["messages"]) for row in rows})
        path = tmp_path / f"{dataset}.jsonl"
        jsonl(path, rows)
        source = fake_repo / "results" / f"{dataset}-source.json"
        source.write_text(json.dumps({"dataset": dataset, "example_text": "DO_NOT_EMBED_SOURCE_DOCUMENT"}))
        candidate_args += ["--candidates", f"{dataset}={path}"]
        source_args += ["--source-manifest", f"{dataset}={source}"]

    tokenizer = Tokenizer()
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return tokenizer
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    monkeypatch.setattr(suite, "verify_pinned_file", lambda *args: {"test_stub": True})
    import full_serving.verify_data as validation
    monkeypatch.setattr(validation, "verify_bundle", lambda *args: {"status": "passed", "requests": 448})
    private, public = tmp_path / "suite-private", tmp_path / "suite-public"
    monkeypatch.setattr(sys, "argv", ["prepare_suite", *candidate_args, *source_args,
        "--wildchat-private", str(wild_private), "--wildchat-manifest", str(wild_manifest),
        "--tokenizer-dir", str(tmp_path / "tokenizer"), "--private-output", str(private),
        "--public-output", str(public)])
    suite.main()
    manifest = json.loads((public / "suite-manifest.json").read_text())
    assert manifest["selection_status"] == "pilot_ready_final_candidates_not_frozen"
    assert manifest.get("final_protocol_frozen") is not True
    assert manifest["datasets_kept_separate"] is True and manifest["pooled_speedup_claim"] is False
    assert [cohort["dataset"] for cohort in manifest["cohorts"]] == ["wildchat", *suite.DATASETS]
    assert "Trace-only" in manifest["burstgpt"]
    assert "DO_NOT_" not in json.dumps(manifest)
    assert "private-wildchat-447" not in json.dumps(manifest)
    all_selected = []
    for dataset in ["wildchat", *suite.DATASETS]:
        rows = suite.load_rows(private / dataset / "requests.jsonl")
        assert len(rows) == 128 and {row["dataset"] for row in rows} == {dataset}
        assert sum(row["split"] == "pilot" for row in rows) == 64
        assert sum(row["split"] == "final" for row in rows) == 64
        assert not banned_ids & {row["id"] for row in rows}
        if dataset != "wildchat":
            assert all(row["messages"] == original_messages[row["id"]] for row in rows)
        all_selected.extend(rows)
        assert stat.S_IMODE((private / dataset / "requests.jsonl").stat().st_mode) == 0o600
    assert len({suite.message_key(row["messages"]) for row in all_selected}) == 640
    assert [row["id"] for row in suite.load_rows(private / "wildchat/requests.jsonl")] == [row["id"] for row in wild[:128]]
    assert {path.name: file_digest(path) for path in wild_private.glob("*.jsonl")} == old_digests
    wild_cohort = manifest["cohorts"][0]
    assert wild_cohort["other_final_bundles"] == "reserved, not discarded or measured"
