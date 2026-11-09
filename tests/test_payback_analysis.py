"""Synthetic timings test inference/contract logic only, never GPU evidence."""
import copy
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

from followup.payback_analysis import (analyze, criterion, estimate, freeze_manifest,
    manifest_hash, pilot_gate, validate_manifest, write_report)


def fixture(split="final", repetitions=2):
    blocks=[f"{split}-{i}" for i in range(8 if split=="final" else 2)]
    m={"payback_schema_version":1,"split":split,
       "matrix":{"arms":["KEEP","PREINSTALLED","CHARGED"],"regimes":["stable","transient","persistent","switching"],
                 "batch_tokens":[128,1024],"horizons":[128,512],"block_ids":blocks,"candidate_ids":["native"],"repetitions":repetitions},
       "analysis":{"primary_horizon":512,"bootstrap_samples":200,"bootstrap_seed":471,"minimum_independent_blocks":8,
                   "useful_gain_ratio":.95,"stable_ratio_limit":1.02},
       "blocks":[{"block_id":b,"source_document_ids":["document-"+b]} for b in blocks],
       "tapes":[{"block_id":b,"regime":r,"batch_tokens":t,"capture_sha256":"a"*64,"replay_sha256":"b"*64}
                for b,r,t in itertools.product(blocks,["stable","transient","persistent","switching"],[128,1024])],
       "required_record_metadata":{"source_restore_verified":True,"candidate_weights_verified":True}}
    m["freeze"]={"sha256":manifest_hash(m)}
    return m


def records(m,stage="screening"):
    rows=[]
    matrix=m["matrix"]
    for block,regime,batch,horizon,candidate,rep,arm in itertools.product(matrix["block_ids"],matrix["regimes"],matrix["batch_tokens"],matrix["horizons"],matrix["candidate_ids"],range(matrix["repetitions"]),matrix["arms"]):
        if m["split"]=="pilot" and rep>0 and (stage=="screening" or regime=="stable" or horizon!=512):continue
        base=1.+rep*.1+int(block.split("-")[-1])*.01
        factor={"KEEP":1.,"PREINSTALLED":.8,"CHARGED":1.01 if regime=="stable" else .9}[arm]
        rows.append({"payback_schema_version":1,"split":m["split"],"config_hash":manifest_hash(m),"block_id":block,
            "regime":regime,"batch_tokens":batch,"horizon_steps":horizon,"candidate_id":candidate,"repetition":rep,"arm":arm,
            "status":"complete","measurement_kind":"gpu_payback","evidence_eligible":True,"correctness_pass":True,
            "steps_completed":horizon,"elapsed_seconds":base*factor,"migration_bytes":100 if arm=="CHARGED" else 0,
            "capture_sha256":"a"*64,"replay_sha256":"b"*64,"source_map_sha256":"c"*64,"candidate_map_sha256":"d"*64,
            "source_weights_sha256":"e"*64,"source_prompt_ids":["document-"+block],"state_id":f"{block}-{regime}-{batch}",
            "source_restore_verified":True,"candidate_weights_verified":True,
            "allocation_id":"synthetic-job", "hostname":"synthetic-host", "process_run_id":"synthetic-run",
            "fixture_notice":"Synthetic unit-test data, not GPU timings"})
    return rows


class PaybackTests(unittest.TestCase):
    def test_complete_paired_ratios_and_independent_units(self):
        m=fixture(); result=analyze(m,records(m))
        self.assertEqual(result["status"],"complete")
        pay=[x for x in result["aggregates"] if x["contrast"]=="payback" and x["group"]=="changing"]
        for row in pay:
            self.assertAlmostEqual(row["ratio"],.9)
            self.assertEqual(row["independent_blocks"],8)
        self.assertEqual(result["verdicts"][0]["verdict"],"useful_charged_payback_demonstrated")
        self.assertFalse(result["verdicts"][0]["permits_final_data_policy_selection"])

    def test_technical_repetitions_do_not_add_independent_blocks(self):
        m=fixture("pilot",3)
        result=analyze(m,records(m,"confirmation"),stage="confirmation")
        self.assertEqual(result["expected_arm_records"],168)
        self.assertTrue(all(c["independent_blocks"]==2 for c in result["cells"]))
        self.assertTrue(all(c["ci95"] is None for c in result["cells"]))
        self.assertTrue(pilot_gate(result,confirmation=True)["candidates"][0]["continue"])

    def test_pilot_screening_and_confirmation_are_distinct(self):
        m=fixture("pilot",3); rows=records(m)
        initial=analyze(m,rows)
        self.assertEqual(initial["expected_arm_records"],96)
        self.assertTrue(pilot_gate(initial)["candidates"][0]["continue"])
        incomplete=analyze(m,rows,stage="confirmation")
        self.assertEqual(incomplete["status"],"incomplete")
        self.assertFalse(pilot_gate(incomplete,confirmation=True)["candidates"][0]["continue"])
        with self.assertRaises(ValueError):pilot_gate(analyze(fixture(),records(fixture())))

    def test_missing_or_corrupt_arms_prevent_confirmatory_verdict(self):
        m=fixture(); rows=records(m)
        incomplete=analyze(m,rows[1:])
        self.assertEqual(incomplete["status"],"incomplete")
        self.assertEqual(incomplete["verdicts"][0]["verdict"],"incomplete")
        rows[1]["candidate_map_sha256"]="f"*64
        corrupted=analyze(m,rows)
        self.assertEqual(len(corrupted["invalid_pairs"]),1)
        self.assertEqual(corrupted["status"],"incomplete")

    def test_wrong_capture_correctness_or_complete_horizon_rejected(self):
        for field,value in [("capture_sha256","f"*64),("correctness_pass",False),("steps_completed",127),
                            ("source_restore_verified",False),("elapsed_seconds",float("nan"))]:
            m=fixture(); rows=records(m);rows[0][field]=value
            self.assertEqual(analyze(m,rows)["status"],"incomplete",field)

    def test_duplicates_and_unrelated_manifests_fail_loudly(self):
        m=fixture();rows=records(m)
        with self.assertRaises(ValueError):analyze(m,rows+[rows[0]])
        rows[0]["config_hash"]="0"*64
        with self.assertRaises(ValueError):analyze(m,rows)

    def test_cross_allocation_arms_cannot_form_a_pair(self):
        m=fixture(); rows=records(m)
        rows[1]["allocation_id"]="another-synthetic-job"
        self.assertEqual(analyze(m,rows)["status"],"incomplete")

    def test_source_document_overlap_is_not_independent(self):
        m=fixture();m["blocks"][1]["source_document_ids"]=m["blocks"][0]["source_document_ids"]
        m["freeze"]["sha256"]=manifest_hash(m)
        with self.assertRaises(ValueError):validate_manifest(m)

    def test_candidate_change_between_horizons_or_repeats_is_rejected(self):
        m=fixture();rows=records(m)
        for row in rows:
            if row["block_id"]=="final-0" and row["regime"]=="stable" and row["batch_tokens"]==128 and row["repetition"]==1:
                row["candidate_map_sha256"]="f"*64
        result=analyze(m,rows)
        self.assertEqual(result["status"],"incomplete")
        self.assertTrue(any("source_candidate_or_replay_changed_between_horizons_or_repetitions" in issue["issues"] for issue in result["invalid_pairs"]))

    def test_margins_and_wide_intervals(self):
        self.assertEqual(criterion([.90,.94],.95),"pass")
        self.assertEqual(criterion([.96,.99],.95),"fail")
        self.assertEqual(criterion([.94,1.01],.95),"inconclusive")
        self.assertEqual(criterion(None,.95),"inconclusive")
        values={str(i):math.log(.8+i*.05) for i in range(8)}
        result=estimate(values,samples=1000,seed=1,minimum_blocks=8,family_size=12)
        self.assertLessEqual(result["family_adjusted_ci"][0],result["ci95"][0])
        self.assertGreaterEqual(result["family_adjusted_ci"][1],result["ci95"][1])

    def test_freeze_disjointness_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pilot=fixture("pilot",3);pilot.pop("freeze")
            source=root/"pilot.json";source.write_text(json.dumps(pilot))
            frozen=root/"pilot-frozen.json";freeze_manifest(source,frozen)
            with self.assertRaises(FileExistsError):freeze_manifest(source,frozen)
            final=fixture();final.pop("freeze");finalsrc=root/"final.json";finalsrc.write_text(json.dumps(final))
            result=freeze_manifest(finalsrc,root/"final-frozen.json",[frozen])
            self.assertEqual(validate_manifest(result),result["freeze"]["sha256"])
            final["blocks"][0]["source_document_ids"]=pilot["blocks"][0]["source_document_ids"]
            finalsrc.write_text(json.dumps(final))
            with self.assertRaises(ValueError):freeze_manifest(finalsrc,root/"bad-final.json",[frozen])

    def test_report_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            dest=Path(folder)/"report";m=fixture();result=analyze(m,records(m))
            write_report(result,dest)
            self.assertTrue((dest/"summary.json").is_file())
            with self.assertRaises(FileExistsError):write_report(result,dest)


if __name__=="__main__":unittest.main()
