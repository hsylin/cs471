"""CPU contract checks: genuine full prompts, explicit limits, held-out identity."""
import json
from pathlib import Path
import stat

import pytest

from full_serving.data import (PriorExclusions, conversation_prefix, digest,
                               load_prior_manifests, prepare_dataset, text_digest)
from full_serving.prepare_data import private_write, serialize_requests


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        # Deliberately include template overhead: limits apply to the complete
        # formatted prompt, not only the user's raw body.
        return list(range(sum(len(m["content"].split()) for m in messages) + 9))


def row(identity, words=20, *, history=False):
    messages = [{"role": "user", "content": " ".join([f"unique-{identity}"] * words)}]
    if history:
        messages += [{"role": "assistant", "content": "This is recorded context."},
                     {"role": "user", "content": f"Please continue the answer for {identity}."}]
    messages += [{"role": "assistant", "content": "DATASET TARGET MUST NOT BE SENT"}]
    return {"conversation_id": identity, "conversation": messages, "language": "English", "toxic": False}


def prepare(rows, prior=None, **kwargs):
    tokenizer = Tokenizer()
    defaults = dict(pilot_requests=4, final_bundles=2, requests_per_bundle=2, smoke_requests=3)
    defaults.update(kwargs)
    requests, manifest = prepare_dataset(rows, tokenizer, prior or PriorExclusions(), **defaults)
    return requests, manifest, tokenizer


def test_full_messages_and_template_preserved_beyond_512_with_actual_next_turn():
    rows = [row(str(i), 900+i, history=True) for i in range(8)]
    requests, manifest, tokenizer = prepare(rows)
    assert len(requests) == 8
    assert min(r["input_tokens"] for r in requests) > 512
    by_id = {r["conversation_id"]: r for r in rows}
    for request in requests:
        assert request["messages"] == by_id[request["conversation_id"]]["conversation"][:-1]
        assert request["messages"][-1]["role"] == "user"
        assert request["recorded_assistant_history_turns"] == 1
        assert "DATASET TARGET" not in json.dumps(request)
    assert all(options == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False,
                               truncation=False, return_dict=False)
               for _, options in tokenizer.calls)
    assert manifest["formatting"]["truncation"] is False
    assert manifest["generation"]["ignore_eos"] is False
    assert all("messages" not in r for r in manifest["requests"])
    assert manifest["final_protocol_frozen"] is False


def test_context_limit_includes_template_and_output_allowance_with_explicit_rejection():
    requests, manifest, _ = prepare([row("overflow", 82)] + [row(str(i), 81) for i in range(8)],
                                   max_context=100, max_new_tokens=10)
    assert all(r["input_tokens"] == 90 for r in requests)
    # Force the overflow candidate to precede accepted candidates by selecting a
    # seed for its ID hash, without any performance observation.
    for seed in range(20):
        _, candidate, _ = prepare([row("overflow", 82)] + [row(str(i), 81) for i in range(8)],
                                  max_context=100, max_new_tokens=10, seed=seed)
        if candidate["exclusion_counts"].get("context_overflow"):
            break
    else:
        pytest.fail("test did not examine over-limit candidate")
    reject = next(r for r in candidate["exclusions"] if r["reason"] == "context_overflow")
    assert reject["input_tokens"] == 91
    assert reject["max_tokens"] == 10


def test_prior_identity_and_normalized_source_and_candidate_duplicates_excluded():
    prior = PriorExclusions(conversation_ids={"used"}, normalized_hashes={text_digest("Already   Seen")})
    old_text = row("different-id")
    old_text["conversation"][0]["content"] = "ＡＬＲＥＡＤＹ seen"
    original = row("first")
    duplicate = row("duplicate")
    duplicate["conversation"][0]["content"] = original["conversation"][0]["content"].upper()
    requests, manifest, _ = prepare([row("used"), old_text, original, duplicate] + [row(str(i)) for i in range(8)], prior)
    assert not {"used", "different-id"} & {r["conversation_id"] for r in requests}
    assert manifest["exclusion_counts"]["prior_conversation_id"] == 1
    assert manifest["exclusion_counts"]["prior_normalized_source_content"] == 1
    assert manifest["exclusion_counts"]["duplicate_candidate_content"] == 1
    assert len({r["first_user_normalized_sha256"] for r in requests}) == len(requests)


def test_hash_selection_is_order_independent_conversation_disjoint_and_smoke_in_pilot():
    rows = [row(str(i), 10+i*100) for i in range(30)]
    first, manifest, _ = prepare(rows)
    reverse, _, _ = prepare(rows[::-1])
    assert [r["id"] for r in first] == [r["id"] for r in reverse]
    pilot = [r for r in first if r["split"] == "pilot"]
    final = [r for r in first if r["split"] == "final"]
    assert len({r["conversation_id"] for r in first}) == 8
    assert not {r["conversation_id"] for r in pilot} & {r["conversation_id"] for r in final}
    assert set(manifest["smoke_request_ids"]) <= {r["id"] for r in pilot}
    smoke_lengths = [r["input_tokens"] for r in pilot if r["id"] in manifest["smoke_request_ids"]]
    assert min(smoke_lengths) == min(r["input_tokens"] for r in pilot)
    assert max(smoke_lengths) == max(r["input_tokens"] for r in pilot)
    assert manifest["counts"]["unexamined_context_eligibility"] == 22


@pytest.mark.parametrize("messages", [
    [{"role": "assistant", "content": "answer first"}],
    [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
    [{"role": "tool", "content": "unsupported tool data"}],
    [{"role": "user", "content": "  "}],
    [{"role": "user", "content": ["non-text"]}],
])
def test_invalid_role_or_content_is_rejected(messages):
    with pytest.raises(ValueError):
        conversation_prefix({"conversation": messages})


def test_system_message_and_user_ending_conversation_supported_without_synthetic_text():
    messages = [{"role": "system", "content": "Original system instruction"},
                {"role": "user", "content": "Original user request"}]
    prefix, index = conversation_prefix({"conversation": messages})
    assert prefix == messages and index == 1


def test_no_reuse_when_frame_cannot_fill_sample():
    with pytest.raises(ValueError, match="no reuse"):
        prepare([row("only")])


def test_public_prior_manifest_seal_required(tmp_path):
    path = tmp_path / "source-manifest.json"
    data = {"blocks": [{"sources": [{"dataset": "wildchat", "source_id": "old", "normalized_text_sha256": text_digest("old text")}]}]}
    data["public_manifest_sha256"] = digest(data)
    path.write_text(json.dumps(data))
    prior = load_prior_manifests([path])
    assert prior.conversation_ids == {"old"}
    assert text_digest("OLD TEXT") in prior.normalized_hashes
    data["blocks"][0]["sources"][0]["source_id"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="seal"):
        load_prior_manifests([path])


def test_private_payloads_owner_only_and_no_overwrite(tmp_path):
    path = tmp_path / "private.jsonl"
    private_write(path, "secret conversation\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        private_write(path, "overwrite")


def test_actual_unicode_and_embedded_newlines_roundtrip_as_single_jsonl_request():
    rows = [{"id": "unicode", "messages": [{"role": "user", "content": "文字\u2028line separator\u2029paragraph\nnewline"}]}]
    text = serialize_requests(rows)
    assert len(text.splitlines()) == 1
    assert [json.loads(line) for line in text.splitlines()] == rows
