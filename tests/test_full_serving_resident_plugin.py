import copy
from importlib.metadata import distributions
from pathlib import Path
from types import SimpleNamespace

import pytest

from full_serving import resident_plugin as p
from full_serving.resident_diagnostic import validate_weight_records


def config():
    return {"schema": p.CONFIG_SCHEMA, "mode": "resident_v1", "vllm_commit": p.PIN,
            "model_revision": p.MODEL_REVISION, "physical_to_logical_map": p.balanced_map(),
            "startup_diagnostic": False}


def test_paired_map_matches_policy_and_checkpoint_slot_ownership():
    from full_serving.resident_policy import balanced_map
    assert p.balanced_map() == balanced_map()
    for rank in range(8):
        assert p.balanced_map()[20 * rank:20 * rank + 16] == list(range(16 * rank, 16 * rank + 16))
        assert p.balanced_map()[20 * rank + 16:20 * rank + 20] == list(range(16 * (rank ^ 1), 16 * (rank ^ 1) + 4))
    assert sorted(p.balanced_map().count(e) for e in range(128)) == [1] * 96 + [2] * 32


def test_changed_or_bool_map_and_unknown_geometry_fail_admission():
    assert p.validate_config(config())
    for key, value in (("physical_to_logical_map", list(range(128)) + list(range(32))),
                       ("vllm_commit", "other"), ("startup_diagnostic", "true")):
        c = config(); c[key] = value
        with pytest.raises(RuntimeError):
            p.validate_config(c)
    c = config(); c["physical_to_logical_map"][1] = True
    with pytest.raises(RuntimeError):
        p.validate_config(c)


def test_prepare_insertion_maps_gathered_ids_and_preserves_all_other_native_results():
    local_ids, gathered_ids, physical_ids = object(), object(), object()
    hidden, scales, metadata, weights = object(), object(), object(), object()
    calls = []
    def native(*args, **kwargs):
        assert args == (local_ids,)
        calls.append("gather")
        return hidden, scales, metadata, gathered_ids, weights
    def mapper(ids):
        assert ids is gathered_ids
        calls.append("map")
        return physical_ids
    result = p._wrap_prepare(native, mapper)(object(), local_ids)
    assert calls == ["gather", "map"]
    assert result == (hidden, scales, metadata, physical_ids, weights)


def test_native_general_plugin_discovery_without_package_install():
    directory = Path(p.__file__).parent / "bootstrap"
    found = [ep for d in distributions(path=[str(directory)]) for ep in d.entry_points]
    assert [(e.group, e.name, e.value) for e in found] == [
        ("vllm.general_plugins", "cs471_resident", "full_serving.resident_plugin:register")]


def weight_records(layers=2):
    mapping = p.balanced_map()
    return [{"rank": rank, "weights": [
        {"layer_index": layer, "physical_slot": slot, "logical_expert": mapping[slot],
         "w13_sha256": f"{layer * 128 + mapping[slot]:064x}",
         "w2_sha256": f"{layer * 128 + mapping[slot] + 1024:064x}"}
        for layer in range(layers) for slot in range(rank * 20, rank * 20 + 20)]}
        for rank in range(8)]


def test_complete_replica_checks_detect_mismatch_missing_rows_and_wrong_ownership():
    records = weight_records()
    assert validate_weight_records(records, p.balanced_map(), layers=2)["duplicated_logical_layer_experts"] == 64
    for change in ("changed_copy", "missing", "owner", "logical"):
        r = copy.deepcopy(records)
        if change == "changed_copy": r[0]["weights"][16]["w13_sha256"] = "f" * 64
        elif change == "missing": r[0]["weights"].pop()
        elif change == "owner": r[0]["weights"][0]["physical_slot"] = 20
        else: r[0]["weights"][0]["logical_expert"] = 7
        with pytest.raises(AssertionError):
            validate_weight_records(r, p.balanced_map(), layers=2)


def test_source_manifest_rejects_changed_or_external_files(tmp_path):
    names = ("resident_plugin.py", "custom_entrypoint.py", "resident_policy.py", "resident_gpu.py", "resident_diagnostic.py")
    paths = ["full_serving/" + n for n in names]
    paths += ["full_serving/bootstrap/cs471_resident-0.1.dist-info/" + n for n in ("entry_points.txt", "METADATA")]
    for name in paths:
        f = tmp_path / name; f.parent.mkdir(parents=True, exist_ok=True); f.write_text(name)
    manifest = {"schema": p.SOURCES_SCHEMA, "files": {n: p.sha(tmp_path / n) for n in paths}}
    p.verify_source_manifest(manifest, tmp_path)
    (tmp_path / paths[0]).write_text("changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        p.verify_source_manifest(manifest, tmp_path)


def test_plugin_registration_is_inactive_without_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("FULL_SERVING_CUSTOM_MODE", raising=False)
    monkeypatch.setattr(p, "load_admission", lambda: (_ for _ in ()).throw(AssertionError("should not read custom config")))
    assert p.register() is None
