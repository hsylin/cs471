"""Synthetic deterministic fixtures; no benchmark observations are created."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from optimization.analysis import (analyze, canonical_hash, freeze_manifest, manifest_hash,
                                    select_configs, validate_manifest)


CONFIGS={"DEFAULT":None,"SMALL":{"BLOCK_SIZE_M":16},"MEDIUM":{"BLOCK_SIZE_M":32}}


def fixture(phase="final"):
    blocks=[f"{phase}-block-{i}" for i in range(8 if phase=="final" else 2)]
    m={"optimization_schema_version":1,"phase":phase,"split":"final" if phase=="final" else "pilot",
       "matrix":{"block_ids":blocks,"batch_tokens":[128,1024],"repetitions":2 if phase=="final" else 1,
                 "regimes":["stable","transient","persistent","switching"],"kernel_variants":["DEFAULT","TUNED"],"policies":["B3","SKIP"]},
       "blocks":[{"block_id":b,"source_document_ids":["doc-"+b]} for b in blocks],
       "trace":{"measured_steps":512,"warmup_steps":32,"raw_steps":544},
       "tuning":{"candidate_configs":CONFIGS,"forwards_per_trial":32,"tie_ratio":1.01,"max_pilot_cell_ratio":1.02},
       "selected_configs":{str(b):{"candidate_id":"SMALL","config":CONFIGS["SMALL"]} for b in [128,1024]},
       "analysis":{"minimum_source_blocks":8,"bootstrap_samples":200,"bootstrap_seed":10471,"useful_gain_ratio":.95,"stable_ratio_limit":1.02,
                   "cell_families":{"changing":6,"stable":2},"interval_scope":"Synthetic fixture only"},
       "tapes":[{"block_id":b,"regime":r,"batch_tokens":t,"capture_sha256":"a"*64,"replay_sha256":"b"*64}
                for b,r,t in itertools.product(blocks,["stable","transient","persistent","switching"],[128,1024])],
       "banks":[{"block_id":b,"capture_sha256":"a"*64} for b in blocks]}
    if phase=="tuning":
        m["matrix"]={"block_ids":blocks,"batch_tokens":[128,1024],"repetitions":3,"candidate_ids":list(CONFIGS),"domains":[0,1]}
    m["freeze"]={"sha256":manifest_hash(m)}
    return m


def common(m,block):
    return {"optimization_schema_version":1,"config_hash":manifest_hash(m),"phase":m["phase"],"split":m["split"],
            "block_id":block,"status":"complete","evidence_eligible":True,"correctness_pass":True,
            "source_prompt_ids":["doc-"+block],"capture_sha256":"a"*64,"replay_sha256":"b"*64,
            "source_map_sha256":"c"*64,"source_weights_sha256":"d"*64,"allocation_id":"synthetic","hostname":"synthetic","process_run_id":"synthetic",
            "fixture_only":"Synthetic unit test; not actual GPU evidence"}


def kernel_fields(identifier):
    return {"kernel_config_id":identifier,"kernel_override_config":CONFIGS[identifier],
            "kernel_config_sha256":canonical_hash(CONFIGS[identifier]),"resolved_kernel_config":CONFIGS[identifier] or {"BLOCK_SIZE_M":64}}


def policy_records(m):
    rows=[]
    x=m["matrix"]
    for block,regime,batch,rep,kernel,policy in itertools.product(x["block_ids"],x["regimes"],x["batch_tokens"],range(x["repetitions"]),x["kernel_variants"],x["policies"]):
        base=10.+int(block[-1])+rep+(policy=="B3")*.1
        ratio=(1.01 if regime=="stable" else .9) if kernel=="TUNED" else 1.
        rows.append({**common(m,block),**kernel_fields("SMALL" if kernel=="TUNED" else "DEFAULT"),
                     "measurement_kind":"gpu_optimization","regime":regime,"batch_tokens":batch,"repetition":rep,"kernel_variant":kernel,"policy":policy,
                     "elapsed_seconds":base*ratio,"steps_completed":512,"observation_warmup_steps":32,"raw_trace_steps":544,
                     "decision_trajectory_sha256":("e" if policy=="B3" else "f")*64})
    return rows


def tune_records(m,small=.9,medium=.95):
    rows=[];x=m["matrix"]
    for block,domain,batch,rep,candidate in itertools.product(x["block_ids"],x["domains"],x["batch_tokens"],range(x["repetitions"]),x["candidate_ids"]):
        base=1.+rep*.01+domain*.1
        rows.append({**common(m,block),**kernel_fields(candidate),"measurement_kind":"gpu_kernel_tuning",
                     "domain":domain,"batch_tokens":batch,"repetition":rep,"candidate_id":candidate,"steps_completed":32,
                     "elapsed_seconds":base*{"DEFAULT":1,"SMALL":small,"MEDIUM":medium}[candidate]})
    return rows


class OptimizationAnalysisTests(unittest.TestCase):
    def test_four_arm_primary_acceptance_and_mechanism_control(self):
        m=fixture();s=analyze(m,policy_records(m))
        self.assertEqual(s["expected_records"],512);self.assertEqual(s["eligible_matched_groups"],128)
        self.assertEqual(s["acceptance"],"pass");self.assertEqual(s["stable_guard"],"pass")
        for row in s["aggregates"]:
            if row["contrast"]=="kernel_B3" and row["group"]=="changing":self.assertAlmostEqual(row["ratio"],.9)
            if row["contrast"]=="interaction":self.assertAlmostEqual(row["ratio"],1.)

    def test_repetitions_are_not_resampling_units(self):
        m=fixture("validation");s=analyze(m,policy_records(m))
        self.assertEqual(s["expected_records"],64)
        self.assertTrue(all(row["source_blocks"]==2 and row["ci95"] is None for row in s["cells"]))
        self.assertFalse(s["claim_eligible"])

    def test_changed_policy_trajectory_invalidates_kernel_attribution(self):
        m=fixture();rows=policy_records(m);rows[0]["decision_trajectory_sha256"]="0"*64
        s=analyze(m,rows)
        self.assertEqual(s["status"],"incomplete");self.assertEqual(s["acceptance"],"inconclusive")
        self.assertIn("B3:kernel_changed_decision_trajectory",s["invalid_groups"][0]["issues"])

    def test_missing_arm_duplicate_and_changed_inputs(self):
        m=fixture();rows=policy_records(m)
        self.assertEqual(analyze(m,rows[1:])["status"],"incomplete")
        with self.assertRaises(ValueError):analyze(m,rows+[rows[0]])
        rows[0]["replay_sha256"]="f"*64
        self.assertEqual(analyze(m,rows)["status"],"incomplete")

    def test_tuning_selects_per_batch_and_keeps_all_candidates(self):
        m=fixture("tuning");rows=tune_records(m)
        for row in rows:
            if row["batch_tokens"]==1024 and row["candidate_id"]=="SMALL":row["elapsed_seconds"]*=1.15
        result=select_configs(m,rows)
        self.assertEqual(result["selected_configs"]["128"]["candidate_id"],"SMALL")
        self.assertEqual(result["selected_configs"]["1024"]["candidate_id"],"MEDIUM")
        self.assertEqual(len(result["candidate_results"]),6)

    def test_default_wins_one_percent_noise_tie(self):
        m=fixture("tuning");result=select_configs(m,tune_records(m,small=.995,medium=1.))
        self.assertTrue(all(c["candidate_id"]=="DEFAULT" for c in result["selected_configs"].values()))

    def test_pilot_cell_guard_rejects_aggregate_winner(self):
        m=fixture("tuning");rows=tune_records(m,small=.7,medium=.96)
        for row in rows:
            if row["candidate_id"]=="SMALL" and row["block_id"].endswith('0') and row["domain"]==0:row["elapsed_seconds"]*=1.5
        result=select_configs(m,rows)
        self.assertTrue(all(c["candidate_id"]=="MEDIUM" for c in result["selected_configs"].values()))

    def test_invalid_candidate_disqualified_missing_trials_no_selection(self):
        m=fixture("tuning");rows=tune_records(m);rows[1]["correctness_pass"]=False
        result=select_configs(m,rows)
        self.assertEqual(result["selected_configs"]["128"]["candidate_id"],"MEDIUM")
        self.assertEqual(select_configs(m,rows[1:])["status"],"incomplete")

    def test_old_document_overlap_rejected(self):
        m=fixture();m["excluded_source_document_ids"]=m["blocks"][0]["source_document_ids"];m["freeze"]["sha256"]=manifest_hash(m)
        with self.assertRaises(ValueError):validate_manifest(m)

    def test_freeze_requires_selection_preserves_null_default(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);m=fixture("tuning");m.pop("freeze");source=root/"tuning.json";source.write_text(json.dumps(m))
            frozen=root/"frozen.json";result=freeze_manifest(source,frozen)
            self.assertEqual(validate_manifest(result),result["freeze"]["sha256"])
            with self.assertRaises(FileExistsError):freeze_manifest(source,frozen)
            validation=fixture("validation");validation.pop("freeze");path=root/"validation.json";path.write_text(json.dumps(validation))
            with self.assertRaises(ValueError):freeze_manifest(path,root/"validation-frozen.json")

    def test_final_freeze_links_selection_to_exact_tuning_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            tuning=fixture("tuning"); tuning_path=root/"tuning-frozen.json"
            tuning_path.write_text(json.dumps(tuning))
            selection={"status":"selected","config_hash":manifest_hash(tuning),
                       "selected_configs":fixture()["selected_configs"]}
            selection_path=root/"selection.json";selection_path.write_text(json.dumps(selection))
            validation=fixture("validation");validation.pop("freeze")
            validation_path=root/"validation.json";validation_path.write_text(json.dumps(validation))
            validation_frozen=root/"validation-frozen.json"
            prior=freeze_manifest(validation_path,validation_frozen,selection=selection_path)
            final=fixture();final.pop("freeze");final_path=root/"final.json";final_path.write_text(json.dumps(final))
            result=freeze_manifest(final_path,root/"final-frozen.json",selection=selection_path,
                                   pilots=[tuning_path,validation_frozen])
            self.assertEqual(result["tuning_manifest_sha256"],manifest_hash(tuning))
            altered_tuning=copy.deepcopy(tuning);altered_tuning["name"]="different pilot"
            altered_tuning["freeze"]["sha256"]=manifest_hash(altered_tuning)
            other_path=root/"other-tuning.json";other_path.write_text(json.dumps(altered_tuning))
            with self.assertRaisesRegex(ValueError,"selection source"):
                freeze_manifest(final_path,root/"invalid-tuning.json",selection=selection_path,
                                pilots=[other_path,validation_frozen])
            for field,value in (("selection_sha256","f"*64),("tuning_manifest_sha256","e"*64),
                                ("selected_configs",{str(b):{"candidate_id":"MEDIUM","config":CONFIGS["MEDIUM"]} for b in [128,1024]})):
                altered=copy.deepcopy(prior);altered[field]=value
                altered["freeze"]["sha256"]=manifest_hash(altered)
                path=root/("mismatch-"+field+".json");path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError,"same frozen pilot selection"):
                    freeze_manifest(final_path,root/("invalid-"+field+".json"),selection=selection_path,
                                    pilots=[tuning_path,path])


if __name__=="__main__":unittest.main()
