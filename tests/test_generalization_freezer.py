"""Mutations of evidence identity and declared calibration boundaries."""
import copy
import json
import sys
import pytest
from scripts import freeze_generalization as freezer


def calibration_fixture():
    plan={'phase':'calibration','content_selection_sha256':'content-v2',
          'execution_source_sha256':'execution-v2',
          'banks':[{'block_id':'wildchat-pilot-01','capture_sha256':'capture-v2'}]}
    starts=[10_000+i*10 for i in range(256)]
    finishes=[t+8 for t in starts]
    elapsed=finishes[-1]-starts[0]
    row={'phase':'calibration','arm':'native','block_id':'wildchat-pilot-01',
         'status':'complete','correctness_pass':True,'resident_state_verified':True,
         'normal_completion_verified':True,'migration_bytes':0,'allocation_id':'4872248',
         'content_selection_sha256':'content-v2','execution_source_sha256':'execution-v2',
         'capture_sha256':'capture-v2','batch_tokens':128,'policy':'SKIP',
         'config_hash':freezer.manifest_hash(plan),'record_id':'record-id',
         'coordinator_start_ns':starts,'coordinator_finish_ns':finishes,
         'calibration_capacity':{'calibration_jobs':256,'calibration_elapsed_ns':elapsed,
                                 'native_mean_cycle_ns':elapsed/256}}
    return plan,row


def invoke(plan,row):
    return freezer.calibration_table([row],{'wildchat-pilot-01'},plan,'content-v2','execution-v2')


def test_valid_calibration_uses_raw_complete_batch_boundary():
    plan,row=calibration_fixture();result=invoke(plan,row)
    assert result[row['block_id']]['native_mean_cycle_ns']==2558/256


def test_old_fulltext_only_selection_with_same_block_ids_is_rejected():
    plan,row=calibration_fixture()
    # Coherently old record and plan, with the very same block identity; a
    # block-ID-only check would accept this ineligible calibration.
    plan['content_selection_sha256']='content-v1-fulltext-only'
    row['content_selection_sha256']=plan['content_selection_sha256']
    row['config_hash']=freezer.manifest_hash(plan)
    with pytest.raises(ValueError,match='Stale or ineligible'):invoke(plan,row)


@pytest.mark.parametrize('field,value',[
    ('content_selection_sha256','content-v1-fulltext-only'),
    ('capture_sha256','wrong-bank'),('execution_source_sha256','stale-execution'),
    ('config_hash','different-calibration-plan'),('batch_tokens',1024),('policy','B3')])
def test_record_identity_mutations_reject(field,value):
    plan,row=calibration_fixture();row[field]=value
    with pytest.raises(ValueError,match='identity changed'):invoke(plan,row)


def test_stale_execution_plan_is_rejected_even_with_matching_record():
    plan,row=calibration_fixture();plan['execution_source_sha256']='execution-v1'
    row['execution_source_sha256']='execution-v1';row['config_hash']=freezer.manifest_hash(plan)
    with pytest.raises(ValueError,match='Stale or ineligible'):invoke(plan,row)


@pytest.mark.parametrize('field,value,match',[
    ('calibration_elapsed_ns',2560,'elapsed boundary'),
    ('native_mean_cycle_ns',10.0,'raw clocks'),
    ('calibration_jobs',255,'Calibration boundary')])
def test_raw_clock_declarations_reject(field,value,match):
    plan,row=calibration_fixture();row['calibration_capacity'][field]=value
    with pytest.raises(ValueError,match=match):invoke(plan,row)


def test_changed_raw_endpoint_without_redeclaring_capacity_rejects():
    plan,row=calibration_fixture();row['coordinator_finish_ns'][-1]+=1
    with pytest.raises(ValueError,match='elapsed boundary'):invoke(plan,row)


def test_incomplete_clock_and_repeated_calibration_not_independent_samples():
    plan,row=calibration_fixture();row['coordinator_start_ns'].pop()
    with pytest.raises(ValueError,match='elapsed boundary'):invoke(plan,row)
    plan,row=calibration_fixture()
    with pytest.raises(ValueError,match='One native calibration'):
        freezer.calibration_table([row,copy.deepcopy(row)],{'wildchat-pilot-01'},plan,'content-v2','execution-v2')


def run_initial_checks(tmp_path,monkeypatch,*,prefix_deduplicated=True,external_changed=False):
    content={'selection':{'captured_prefix_deduplicated':prefix_deduplicated},'blocks':[]}
    content['public_manifest_sha256']=freezer.canonical_hash(content)
    external={'blocks':[],'selection':{'seed':471502}}
    external['content_sha256']=freezer.canonical_hash(external)
    if external_changed:external['selection']['seed']=42
    cp=tmp_path/'content.json';ap=tmp_path/'arrivals.json'
    cp.write_text(json.dumps(content));ap.write_text(json.dumps(external))
    monkeypatch.setattr(sys,'argv',['freeze_generalization.py','--phase','calibration','--split','pilot',
        '--capture-root',str(tmp_path/'unused-capture'),'--content-manifest',str(cp),
        '--arrivals',str(ap),'--output',str(tmp_path/'must-not-be-created.json')])
    freezer.main()


def test_fulltext_only_public_selection_cannot_be_frozen(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='Full-text-only split is ineligible'):
        run_initial_checks(tmp_path,monkeypatch,prefix_deduplicated=False)
    assert not (tmp_path/'must-not-be-created.json').exists()


def test_external_manifest_mutation_rejected_before_loading_captures(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='Changed external arrival manifest'):
        run_initial_checks(tmp_path,monkeypatch,external_changed=True)
    assert not (tmp_path/'must-not-be-created.json').exists()
