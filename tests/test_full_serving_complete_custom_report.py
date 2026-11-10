import copy
import json

import pytest

from full_serving import complete_custom_report as complete
from test_full_serving_custom_report import write_experiment


def evidence(tmp_path, monkeypatch):
    root, _ = write_experiment(tmp_path, monkeypatch)
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    for name, data in (("completed.txt", "controller_exit=0\n"), ("controller.log", ""), ("controller.err", "")):
        (wrapper / name).write_text(data)
    observed = complete.inventory(root, wrapper)
    remote = {"schema": "cs471_stable_remote_inventory_v1", "before": observed, "after": copy.deepcopy(observed)}
    monkeypatch.setattr(complete, "verify_source", lambda *args: "a" * 40)
    return root, wrapper, remote


def test_completed_collection_reconstructs_both_reports_and_receipt(tmp_path, monkeypatch):
    root, wrapper, remote = evidence(tmp_path, monkeypatch)
    result = complete.finish(root, wrapper, remote, tmp_path, tmp_path / "outputs" / "comparison", tmp_path / "audit")
    assert result["status"] == "completed_and_audited"
    assert result["measured_sessions"] == 60
    assert result["automatic_superiority_claim"] is False
    audit = json.loads((tmp_path / "audit" / "independent-statistics-audit.json").read_text())
    assert audit["status"] == "passed" and audit["reconstructed_sessions"] == 60


@pytest.mark.parametrize("corruption", ["wrapper_failed", "partial_sessions", "late_remote_tail", "local_tampering"])
def test_refuses_premature_or_nonidentical_raw_evidence(tmp_path, monkeypatch, corruption):
    root, wrapper, remote = evidence(tmp_path, monkeypatch)
    if corruption == "wrapper_failed":
        (wrapper / "completed.txt").write_text("controller_exit=1\n")
    elif corruption == "partial_sessions":
        state = json.loads((root / "status.json").read_text())
        state["completed_sessions"].pop()
        (root / "status.json").write_text(json.dumps(state))
    elif corruption == "late_remote_tail":
        remote["after"]["controller.err"]["bytes"] += 1
    else:
        (wrapper / "controller.err").write_text("late shutdown warning\n")
    with pytest.raises(ValueError):
        complete.finish(root, wrapper, remote, tmp_path, tmp_path / "outputs" / "comparison", tmp_path / "audit")
    assert not (tmp_path / "audit" / "completion.json").exists()


def test_arithmetic_failure_never_publishes_completion(tmp_path, monkeypatch):
    root, wrapper, remote = evidence(tmp_path, monkeypatch)
    monkeypatch.setattr(complete.audit_final_statistics, "audit", lambda *args: {"status": "failed", "differences": ["changed timing"]})
    with pytest.raises(ValueError, match="independent raw-record"):
        complete.finish(root, wrapper, remote, tmp_path, tmp_path / "outputs" / "comparison", tmp_path / "audit")
    assert not (tmp_path / "audit" / "completion.json").exists()
