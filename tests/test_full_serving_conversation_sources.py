import csv
import json

import pytest

from full_serving import conversation_sources as sources


def test_streaming_released_array_handles_fragmented_unicode_and_multiline_values(tmp_path):
    rows = [{"id": "one_0", "text": "你好\nline\u2028separator"}, {"id": "two_1", "text": "x" * 100}]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    assert list(sources.iter_json_array(path, chunk_size=7)) == rows
    path.write_text('[{"id": "one"},]')
    with pytest.raises(ValueError, match="Trailing array comma"):
        list(sources.iter_json_array(path, chunk_size=3))


def test_sharegpt_retains_authentic_history_and_groups_upstream_segment_ids():
    row = {"id": "original_17", "conversations": [
        {"from": "gpt", "value": "Retained assistant context."},
        {"from": "human", "value": "Actual full final user prompt.\u2028"},
        {"from": "gpt", "value": "Held-out recorded answer."}]}
    converted = sources.sharegpt_candidate(row, 5)
    assert converted["messages"] == [
        {"role": "assistant", "content": "Retained assistant context."},
        {"role": "user", "content": "Actual full final user prompt.\u2028"}]
    assert "Held-out recorded answer" not in json.dumps(converted)
    assert converted["group_id"] == "sharegpt:original"
    assert converted["id"] == "sharegpt:original_17:5"
    assert converted["released_segment_starts_with_assistant"] is True
    assert converted["excluded_target_messages"] == 1
    row["id"] = "original_0"
    assert sources.sharegpt_candidate(row, 7)["group_id"] == converted["group_id"]


def test_burstgpt_keeps_contiguous_failures_lengths_and_absolute_offsets(tmp_path, monkeypatch):
    path = tmp_path / "BurstGPT.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"])
        writer.writeheader()
        for index in range(100):
            output = 0 if index % 2 == 0 else 10
            writer.writerow({"Timestamp": str(index // 2), "Model": "ChatGPT", "Request tokens": 100,
                             "Response tokens": output, "Total tokens": 100 + output, "Log Type": "API log"})
    monkeypatch.setattr(sources, "BURSTGPT_SHA256", sources.sha_file(path))
    block = sources.prepare_burstgpt(path)
    start = block["source_start_row_index"]
    assert [r["source_row_index"] for r in block["rows"]] == list(range(start, start + 24))
    assert block["source_failure_rows_retained"] == 12
    assert block["original_offsets_ns"] == [((index // 2) - (start // 2)) * 1_000_000_000 for index in range(start, start + 24)]
    assert block["kind"] == "metadata_only_no_prompt_text"
    assert block["source_lengths_are_qwen_token_counts"] is False
    assert block["source_failures_are_injected_into_new_server"] is False
    assert all("messages" not in row for row in block["rows"])
    assert sources.prepare_burstgpt(path) == block


def test_source_checksums_fail_closed(tmp_path):
    path = tmp_path / "wrong-source"
    path.write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        sources.prepare_sharegpt(path, tmp_path / "private.jsonl")
    with pytest.raises(ValueError, match="checksum"):
        sources.prepare_burstgpt(path)
    assert not (tmp_path / "private.jsonl").exists()
