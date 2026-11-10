import json

import pytest

from full_serving.code_math_sources import input_messages, iter_rows, select_candidates


def test_instructcoder_uses_only_instruction_and_input_verbatim():
    source = {"instruction": " Change names.\n", "input": "def π():\n    return 1\n",
              "output": "REFERENCE_MUST_NOT_APPEAR"}
    messages = input_messages("instructcoder", source)
    assert messages == [{"role": "user", "content": " Change names.\n\n\nInput code:\ndef π():\n    return 1\n"}]
    source["output"] = "changed reference"
    assert input_messages("instructcoder", source) == messages


def test_numinamath_never_uses_reference_answer_or_cot():
    source = {"problem": "Prove x > 0.", "solution": "secret proof", "answer": "secret answer",
              "messages": [{"role": "assistant", "content": "secret CoT"}], "ref_solution": "secret"}
    assert input_messages("numinamath", source) == [{"role": "user", "content": "Prove x > 0."}]


@pytest.mark.parametrize("dataset,row", [
    ("instructcoder", {"instruction": "", "input": "code"}),
    ("instructcoder", {"instruction": "Do it", "input": None}),
    ("numinamath", {"problem": "   ", "solution": "not a substitute"}),
])
def test_missing_prompt_fields_are_not_filled_with_answers(dataset, row):
    with pytest.raises(ValueError):
        input_messages(dataset, row)


def test_frame_selection_is_independent_of_lengths_outcomes_and_input_order():
    rows = [(str(i), {"problem": "q" * (i + 1), "answer": "x"}) for i in range(20)]
    selected, count = select_candidates("numinamath", rows, count=5, seed=13)
    changed = [(key, {"problem": "completely different", "latency": 12345}) for key, _ in reversed(rows)]
    other, _ = select_candidates("numinamath", changed, count=5, seed=13)
    assert [r["source_id"] for r in selected] == [r["source_id"] for r in other]
    assert count["eligible_rows"] == 20
    assert all(r["source_split"] == "train" for r in selected)
    assert all(r["messages"][-1]["role"] == "user" for r in selected)


def test_exclusions_are_counted_and_duplicate_source_ids_rejected():
    rows = [("a", {"problem": "A?"}), ("b", {"problem": None})]
    _, report = select_candidates("numinamath", rows, count=1)
    assert report["schema_exclusions"] == {"missing_or_empty_problem": 1}
    with pytest.raises(ValueError, match="Duplicate"):
        select_candidates("numinamath", [rows[0], rows[0]], count=1)


def test_instructcoder_row_ids_are_exact_source_positions(tmp_path):
    source = tmp_path / "instructcoder"
    source.mkdir()
    source.joinpath("train.json").write_text(json.dumps([
        {"instruction": "A", "input": "B", "output": "C"},
        {"instruction": "D", "input": "E", "output": "F"},
    ]))
    assert [key for key, _ in iter_rows("instructcoder", tmp_path)] == ["train.json:0", "train.json:1"]
