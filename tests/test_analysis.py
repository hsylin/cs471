"""Synthetic deterministic fixtures test statistics, never benchmark evidence."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from skewbench.analysis import (analyze, analyze_branches, freeze_manifest, manifest_hash,
                               paired_cluster_estimate, verify_frozen_manifest,
                               write_report, write_branch_report, select_cooldown)


def fixture_manifest(split="final"):
    manifest = {
        "schema_version": 1, "split": split,
        "environment": {"model_id": "synthetic-unit-test", "model_revision": "fixture",
                        "hardware": "synthetic-unit-test", "vllm_commit": "fixture"},
        "matrix": {"methods": ["B3", "B4", "B5"],
                   "cases": [{"regime": r, "trace_variant": "core", "phase_steps": 0}
                             for r in ["stationary", "burst"]],
                   "batch_tokens": [128, 1024], "seeds": [1001, 1002, 1003, 1004, 1005],
                   "repetitions": 3},
        "analysis": {"candidate": "B4", "references": ["B3", "B5"],
                     "minimum_independent_seeds": 5, "bootstrap_samples": 200,
                     "regime_ratio_limits": {"burst": .95, "stationary": 1.02},
                     "required_cell_regimes": ["stationary"]}}
    manifest["freeze"] = {"sha256": manifest_hash(manifest)}
    return manifest


def fixture_records(manifest):
    """Mock GPU records to exercise validation; these are not real timings."""
    records = []
    matrix = manifest["matrix"]
    for method in matrix["methods"]:
        for case in matrix["cases"]:
            for batch in matrix["batch_tokens"]:
                for seed in matrix["seeds"]:
                    for repetition in range(matrix["repetitions"]):
                        # Different baseline run duration; exact paired ratios.
                        baseline = 10 + (seed - 1000) + repetition
                        ratio = (.9 if case["regime"] == "burst" else 1.01) if method == "B4" else 1.
                        records.append({"schema_version": 1, "method": method,
                            "regime": case["regime"], "trace_variant": case["trace_variant"],
                            "batch_tokens": batch, "seed": seed, "repetition": repetition,
                            "config_hash": manifest_hash(manifest), "split": manifest["split"],
                            "status": "complete", "correctness_pass": True,
                            "evidence_eligible": True,
                            "measurement_kind": "gpu", "elapsed_seconds": baseline * ratio,
                            "migration_bytes": 100, "accepted": 1, "rejected": 0,
                            "fixture_notice": "synthetic unit test only; not measured GPU evidence"})
    return records


def strict_fixtures():
    manifest = fixture_manifest()
    manifest["environment"].update(layer_id=0, world_size=2, dtype="bfloat16",
        dispatch_backend="replicated_input_allreduce", eplb_mode="synchronous", hidden_size=32, intermediate_size=8)
    manifest["trace"] = {"num_experts": 8, "top_k": 2, "num_steps": 128, "window_steps": 8}
    manifest["policy"] = {"rebalance_interval": 32, "epsilon": .1, "tau": .1, "cooldown_steps": 64, "replica_budget": 2}
    manifest["analysis"]["require_execution_metadata"] = True
    manifest["freeze"]["sha256"] = manifest_hash(manifest)
    records = fixture_records(manifest)
    for record in records:
        record.update(native_vllm_commit="fixture", world_size=2, dtype="bfloat16",
                      dispatch_backend="replicated_input_allreduce", eplb_mode="synchronous",
                      steps_completed=128, measured_horizon_steps=128, observation_warmup_steps=16,
                      raw_trace_steps=144, policy_config=manifest["policy"], trace_sha256="a"*64,
                      weights_source={"source_model": "synthetic-unit-test", "source_revision": "fixture",
                        "layer": 0, "weights_kind": "pretrained", "dtype": "bfloat16", "num_experts": 8,
                        "top_k": 2, "hidden_size": 32, "intermediate_size": 8, "weights_sha256": "b"*64},
                      trace_config={"num_experts": 8, "top_k": 2, "num_steps": 144, "window_steps": 8,
                        "warmup_steps": 16, "phase_steps": 0, "seed": record["seed"],
                        "regime": record["regime"], "tokens_per_step": record["batch_tokens"]},
                      accepted=3, rejected=0, decision_log=[{"step": s} for s in [31,63,95]])
    return manifest, records


def branch_fixtures():
    manifest = fixture_manifest()
    manifest["diagnostics"] = {"horizon_steps": 128, "decision_steps": [63], "analysis_seeds": [1001]}
    manifest["freeze"]["sha256"] = manifest_hash(manifest)
    hashes = {"source_map_sha256": "a"*64, "source_weights_sha256": "b"*64,
              "candidate_map_sha256": "c"*64, "trace_sha256": "d"*64}
    records = []
    for regime in ["stationary", "burst", "persistent", "periodic"]:
        for batch in [128, 1024]:
            records.append({"schema_version": 1, "config_hash": manifest_hash(manifest), "split": "final",
                            "measurement_kind": "gpu_branch", "status": "complete", "correctness_pass": True,
                            "evidence_eligible": True, "future_steps": 128, "decision_step": 63,
                            "regime": regime, "trace_variant": "core", "batch_tokens": batch,
                            "seed": 1001, "repetition": 0, **hashes,
                            "b3_would_allow": True, "gate_decision": {"allowed": False, "tv": .3, "predicted_gain": .2},
                            "branches": [{"branch": b, "elapsed_seconds": t, "migration_bytes": n,
                                          "correctness_pass": True, "correctness_max_abs_error": .001,
                                          "future_steps":128, **hashes}
                                         for b,t,n in [("allow", .75, 4096), ("skip", 1., 0)]]})
    return manifest, records


class AnalysisTests(unittest.TestCase):
    def test_cooldown_selection_is_pilot_only_and_uses_declared_tie_rule(self):
        runs = []
        for cooldown, ratio in [(64, .9), (128, .8), (256, .805)]:
            manifest = fixture_manifest("pilot")
            manifest["policy"] = {"cooldown_steps": cooldown}
            manifest["pilot_selection"] = {"cooldown_steps_grid": [64,128,256]}
            manifest["freeze"]["sha256"] = manifest_hash(manifest)
            records = fixture_records(manifest)
            for record in records:
                if record["method"] == "B5": record["elapsed_seconds"] *= ratio
            runs.append((manifest, records))
        result = select_cooldown(runs)
        self.assertEqual(result["selected_cooldown_steps"], 256)
        self.assertTrue(result["stationary_constraint_met"])
        self.assertEqual(select_cooldown(runs[:2])["status"], "not_evaluable")
        manifest, records = runs[0]
        manifest["split"] = "final"
        manifest["freeze"]["sha256"] = manifest_hash(manifest)
        with self.assertRaisesRegex(ValueError, "never consume final"):
            select_cooldown([(manifest, records)])

    def test_execution_metadata_and_completed_horizon_are_checked(self):
        manifest, records = strict_fixtures()
        self.assertEqual(analyze(records, manifest)["overall_verdict"], "pass")
        records[0]["steps_completed"] = 127
        result = analyze(records, manifest)
        self.assertEqual(result["overall_verdict"], "not_evaluable")
        self.assertIn("steps_completed_mismatch", result["invalid_records"][0]["contract_issues"])

    def test_equal_seed_with_different_trace_hash_is_not_paired(self):
        manifest, records = strict_fixtures()
        records[0]["trace_sha256"] = "c"*64
        result = analyze(records, manifest)
        self.assertEqual(result["overall_verdict"], "not_evaluable")
        self.assertEqual(len(result["invalid_records"]), 3)

    def test_terminal_proposal_is_not_charged_to_empty_future(self):
        manifest, records = strict_fixtures()
        records[0]["decision_log"].append({"step": 127})
        result = analyze(records, manifest)
        self.assertEqual(result["overall_verdict"], "not_evaluable")
        self.assertIn("scheduled_decision_steps_mismatch", result["invalid_records"][0]["contract_issues"])

    def test_branch_payback_is_descriptive_and_signed(self):
        manifest, records = branch_fixtures()
        result = analyze_branches(records, manifest)
        self.assertEqual(result["complete_pairs"], 8)
        self.assertTrue(all(r["payback_seconds"] == .25 for r in result["rows"]))
        self.assertTrue(all(r["b4_allowed"] is False for r in result["rows"]))
        self.assertNotIn("confidence_interval", result)
        with tempfile.TemporaryDirectory() as folder:
            write_branch_report(result, Path(folder))
            self.assertTrue((Path(folder) / "branches.csv").exists())

    def test_branch_pair_must_share_snapshot_and_horizon(self):
        for mutation in ("source_weights_sha256", "source_map_sha256", "trace_sha256", "candidate_map_sha256"):
            manifest, records = branch_fixtures()
            records[0]["branches"][0][mutation] = "f"*64
            result = analyze_branches(records, manifest)
            self.assertEqual(result["complete_pairs"], 7)
            self.assertIsNone(next(r for r in result["rows"] if r["status"] == "invalid")["payback_seconds"])
        manifest, records = branch_fixtures()
        records[0]["future_steps"] = 256
        self.assertEqual(analyze_branches(records, manifest)["complete_pairs"], 7)

    def test_known_paired_ratios_and_independent_unit(self):
        manifest = fixture_manifest()
        result = analyze(fixture_records(manifest), manifest)
        self.assertEqual(result["overall_verdict"], "pass")
        for cell in result["cells"]:
            self.assertEqual(cell["independent_seeds"], 5)
            self.assertEqual(cell["paired_runs"], 15)
            self.assertAlmostEqual(cell["ratio"], .9 if cell["regime"] == "burst" else 1.01)

    def test_bootstrap_is_deterministic_and_seed_based(self):
        logs = {i: math.log(.8 + i * .03) for i in range(8)}
        a = paired_cluster_estimate(logs, bootstrap_samples=500, bootstrap_seed=7)
        b = paired_cluster_estimate(dict(reversed(list(logs.items()))), bootstrap_samples=500, bootstrap_seed=7)
        self.assertEqual(a, b)
        self.assertEqual(a["independent_seeds"], 8)
        self.assertLess(a["ci95"][0], a["ratio"])
        self.assertGreater(a["ci95"][1], a["ratio"])

    def test_missing_run_prevents_confirmatory_verdict(self):
        manifest = fixture_manifest()
        records = fixture_records(manifest)
        records.pop()
        result = analyze(records, manifest)
        self.assertEqual(result["overall_verdict"], "not_evaluable")
        self.assertEqual(len(result["missing_records"]), 1)
        self.assertTrue(any(c["status"] == "incomplete" for c in result["cells"]))

    def test_absent_data_does_not_make_speedup(self):
        result = analyze([], fixture_manifest())
        self.assertEqual(result["overall_verdict"], "not_evaluable")
        self.assertTrue(all(c["ratio"] is None and c["ci95"] is None for c in result["cells"]))

    def test_cpu_and_wrong_outputs_never_count_as_evidence(self):
        manifest = fixture_manifest()
        for override in ({"measurement_kind": "cpu"}, {"correctness_pass": False},
                         {"evidence_eligible": False}, {"measurement_kind": "gpu_smoke"},
                         {"status": "incomplete"}, {"elapsed_seconds": float("nan")}):
            records = fixture_records(manifest)
            records[0].update(override)
            result = analyze(records, manifest)
            self.assertEqual(result["overall_verdict"], "not_evaluable")
            self.assertEqual(len(result["invalid_records"]), 1)

    def test_duplicate_pair_is_rejected(self):
        manifest = fixture_manifest()
        records = fixture_records(manifest)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze(records + [records[0]], manifest)

    def test_pilot_is_exploratory_even_with_exact_gain(self):
        manifest = fixture_manifest("pilot")
        result = analyze(fixture_records(manifest), manifest)
        self.assertEqual(result["overall_verdict"], "not_evaluable")
        self.assertTrue(all(a["verdict"] == "not_evaluable" for a in result["regime_aggregates"]))

    def test_stable_cell_regression_cannot_be_hidden_by_aggregate(self):
        manifest = fixture_manifest()
        records = fixture_records(manifest)
        for record in records:
            if record["method"] == "B4" and record["regime"] == "stationary":
                record["elapsed_seconds"] *= (1.04 if record["batch_tokens"] == 128 else .96) / 1.01
        result = analyze(records, manifest)
        self.assertEqual(result["overall_verdict"], "fail")
        self.assertTrue(all(a["verdict"] == "pass" for a in result["regime_aggregates"]))

    def test_stable_b5_is_descriptive_when_not_predeclared(self):
        manifest = fixture_manifest()
        manifest["analysis"]["regime_references"] = {"burst": ["B3", "B5"], "stationary": ["B3"]}
        manifest["freeze"]["sha256"] = manifest_hash(manifest)
        records = fixture_records(manifest)
        for record in records:
            if record["method"] == "B5" and record["regime"] == "stationary":
                record["elapsed_seconds"] *= .5
        result = analyze(records, manifest)
        self.assertEqual(result["overall_verdict"], "pass")
        descriptive = next(a for a in result["regime_aggregates"]
                           if a["reference"] == "B5" and a["regime"] == "stationary")
        self.assertIsNone(descriptive["ratio_limit"])
        self.assertGreater(descriptive["ratio"], 2)

    def test_intervals_crossing_target_are_inconclusive(self):
        manifest = fixture_manifest()
        records = fixture_records(manifest)
        ratios = [.8, .9, 1., 1.05, 1.1]
        for record in records:
            if record["method"] == "B4" and record["regime"] == "burst":
                record["elapsed_seconds"] *= ratios[record["seed"]-1001] / .9
        result = analyze(records, manifest)
        self.assertEqual(result["overall_verdict"], "inconclusive")

    def test_phase_variants_are_distinct_pairs(self):
        manifest = fixture_manifest()
        manifest["matrix"]["cases"].append({"regime": "burst", "trace_variant": "phase_shift", "phase_steps": 4})
        manifest["freeze"]["sha256"] = manifest_hash(manifest)
        result = analyze(fixture_records(manifest), manifest)
        self.assertEqual(result["overall_verdict"], "pass")
        self.assertEqual(len(result["cells"]), 12)

    def test_too_few_seeds_has_no_interval(self):
        estimate = paired_cluster_estimate({1: math.log(.5), 2: math.log(.5)}, bootstrap_samples=200)
        self.assertAlmostEqual(estimate["ratio"], .5)
        self.assertIsNone(estimate["ci95"])
        self.assertEqual(estimate["status"], "insufficient_seeds")

    def test_freeze_write_once_integrity_and_disjoint_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pilot = fixture_manifest("pilot")
            del pilot["freeze"]
            pilot["matrix"]["seeds"] = [11, 12]
            source, frozen = root / "pilot.json", root / "pilot-frozen.json"
            source.write_text(json.dumps(pilot))
            locked = freeze_manifest(source, frozen)
            self.assertEqual(verify_frozen_manifest(locked), locked["freeze"]["sha256"])
            with self.assertRaises(FileExistsError):
                freeze_manifest(source, frozen)
            edited = copy.deepcopy(locked)
            edited["matrix"]["seeds"].append(13)
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_frozen_manifest(edited)
            final = fixture_manifest()
            del final["freeze"]
            source.write_text(json.dumps(final))
            final_frozen = freeze_manifest(source, root / "final.json", frozen)
            self.assertEqual(final_frozen["pilot_manifest_sha256"], locked["freeze"]["sha256"])
            final["matrix"]["seeds"] = [11]
            source.write_text(json.dumps(final))
            with self.assertRaisesRegex(ValueError, "overlap"):
                freeze_manifest(source, root / "bad.json", frozen)

    def test_report_preserves_missing_values(self):
        with tempfile.TemporaryDirectory() as folder:
            result = analyze([], fixture_manifest())
            write_report(result, Path(folder), plots=False)
            saved = json.loads((Path(folder) / "summary.json").read_text())
            self.assertIsNone(saved["cells"][0]["ratio"])
            self.assertTrue((Path(folder) / "cells.csv").exists())


if __name__ == "__main__":
    unittest.main()
