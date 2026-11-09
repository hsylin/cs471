"""CPU tests of provenance, semantic pairing, and constructed-replay boundaries."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from followup.router_tapes import (MODEL, REVISION, compose_router_tape, load_router_bank,
                                  load_router_tape, save_router_bank)


@pytest.fixture
def bank(tmp_path):
    n, h, k = 32, 4, 2
    arrays = {"hidden_states_bf16": np.arange(n*h, dtype=np.uint16).reshape(n,h),
              "expert_ids": np.tile(np.array([[0, 1]], np.int64), (n,1)),
              "combining_weights": np.tile(np.array([[0.6015625, 0.3984375]], np.float32),(n,1)),
              "token_ids": np.arange(n,dtype=np.int64), "prompt_index": np.repeat([0,1],n//2).astype(np.int32),
              "token_position": np.tile(np.arange(n//2,dtype=np.int32),2),
              "domain_ids": np.repeat([0,1],n//2).astype(np.int8)}
    metadata = {"schema":"qwen_router_bank_v1", "source_model":MODEL, "source_revision":REVISION,
                "layer":0,"top_k":k,"hidden_size":h,"num_experts":8,"block_id":"pilot-01",
                "split":"pilot","source_prompt_ids":["doc-a","doc-b"]}
    return save_router_bank(tmp_path/"banks/bank.npz",metadata,arrays)


def make(bank, path, regime, **kw):
    return compose_router_tape(bank.path,path,regime=regime,batch_tokens=8,num_steps=672,
                               warmup_steps=32,phase_steps=120,duration_steps=16,repeat_steps=1024,**kw)


def test_real_records_preserved_and_reuse_accounted(bank,tmp_path):
    tape=make(bank,tmp_path/"tapes/stable.npz","stable")
    assert np.array_equal(tape.expert_ids,bank.expert_ids[tape.indices])
    assert np.array_equal(tape.combining_weights,bank.combining_weights[tape.indices])
    assert tape.metadata["capture_sha256"]==bank.sha256
    assert sum(tape.metadata["token_use_counts"])==672*8
    assert tape.metadata["natural_temporal_demand"] is False
    assert tape.metadata["unique_captured_records_used"]==16
    assert not tape.indices.flags.writeable
    assert not tape.bank.hidden_states_bf16.flags.writeable
    assert all(len(set(row))==8 for row in tape.indices)


def test_same_history_distinct_future_at_second_decision(bank,tmp_path):
    transient=make(bank,tmp_path/"t.npz","transient")
    persistent=make(bank,tmp_path/"p.npz","persistent")
    assert np.array_equal(transient.indices[:168],persistent.indices[:168])
    assert not np.array_equal(transient.indices[168],persistent.indices[168])
    assert np.all(bank.domain_ids[transient.indices[152:168]]==1)
    assert np.all(bank.domain_ids[transient.indices[:152]]==0)
    assert np.all(bank.domain_ids[transient.indices[168:]]==0)
    assert sum(transient.metadata["domain_schedule"])==16
    # raw159 = warm32 + measured127; complete shared B3 source history.
    assert np.array_equal(transient.indices[:160],persistent.indices[:160])


def test_switching_window_ends_exactly_at_proposal(bank,tmp_path):
    t=compose_router_tape(bank.path,tmp_path/"s.npz",regime="switching",batch_tokens=8,
                          num_steps=672,warmup_steps=32,phase_steps=112,duration_steps=16)
    assert np.all(bank.domain_ids[t.indices[144:160]]==1)
    assert np.all(bank.domain_ids[t.indices[160:176]]==0)


def test_seed_reproducibility_and_alias(bank,tmp_path):
    a=make(bank,tmp_path/"a.npz","transient",seed=99)
    b=make(bank,tmp_path/"b.npz","burst",seed=99)
    assert np.array_equal(a.indices,b.indices)
    c=make(bank,tmp_path/"c.npz","burst",seed=100)
    assert not np.array_equal(a.indices,c.indices)


def test_no_overwrite(bank,tmp_path):
    path=tmp_path/"x.npz";make(bank,path,"stable")
    with pytest.raises(FileExistsError): make(bank,path,"stable")


def test_tampered_bank_rejected(bank):
    with np.load(bank.path) as z: data={k:z[k].copy() for k in z.files}
    data["expert_ids"][0,0]=4
    np.savez_compressed(bank.path,**data)
    with pytest.raises(ValueError,match="content hash"):load_router_bank(bank.path)


def test_tampered_replay_rejected(bank,tmp_path):
    tape=make(bank,tmp_path/"a.npz","stable")
    with np.load(tape.path) as z: data={k:z[k].copy() for k in z.files}
    data["indices"][0,0]=30
    np.savez_compressed(tape.path,**data)
    with pytest.raises(ValueError,match="content hash"):load_router_tape(tape.path)


def test_invalid_bank_semantics(bank,tmp_path):
    from dataclasses import asdict
    arrays={k:getattr(bank,k).copy() for k in ("hidden_states_bf16","expert_ids","combining_weights",
                 "token_ids","prompt_index","token_position","domain_ids")}
    arrays["expert_ids"][0]=[1,1]
    with pytest.raises(ValueError,match="Repeated logical"):save_router_bank(tmp_path/"bad.npz",bank.metadata,arrays)


def test_capture_source_provenance_and_split_integrity():
    script=Path(__file__).resolve().parents[1]/"scripts/capture_router_tapes.py"
    spec=importlib.util.spec_from_file_location("capture_router_tapes",script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    directory=script.parents[1]/"followup/corpus"
    manifest=json.loads((directory/"sources.json").read_text())
    module.validate_source_manifest(manifest)
    for block in manifest["blocks"]:
        for source in block["sources"]:
            text=module.source_text(source,directory)
            assert len(text)>5000
            if source["domain"]=="prose": assert "*** START OF" not in text
    manifest["blocks"][1]["sources"][0]["source_id"]=manifest["blocks"][0]["sources"][0]["source_id"]
    with pytest.raises(ValueError,match="overlap"):module.validate_source_manifest(manifest)


@pytest.mark.parametrize("kwargs",[{"batch_tokens":17},{"num_steps":0},{"seed":-1},{"repeat_steps":16}])
def test_invalid_replay_parameters(bank,tmp_path,kwargs):
    parameters=dict(regime="stable",batch_tokens=8)
    parameters.update(kwargs)
    with pytest.raises(ValueError):compose_router_tape(bank.path,tmp_path/"bad.npz",**parameters)


def test_default_is_one_transient_and_v1_is_immutable(bank,tmp_path):
    old=compose_router_tape(bank.path,tmp_path/"v1.npz",regime="transient",batch_tokens=8,repeat_steps=128)
    before=old.path.read_bytes()
    new=compose_router_tape(bank.path,tmp_path/"v2.npz",regime="transient",batch_tokens=8)
    assert sum(old.metadata["domain_schedule"])>16
    assert sum(new.metadata["domain_schedule"])==16
    assert old.path.read_bytes()==before
    assert old.bank.sha256==new.bank.sha256==bank.sha256


def test_composer_refuses_nonempty_output_before_mutation(tmp_path):
    script=Path(__file__).resolve().parents[1]/"scripts/compose_router_tapes.py"
    spec=importlib.util.spec_from_file_location("compose_router_tapes",script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=tmp_path/"v1";out.mkdir();marker=out/"existing";marker.write_text("preserve")
    with pytest.raises(FileExistsError):module.compose_directory(tmp_path/"banks",out)
    assert marker.read_text()=="preserve"
