"""Synthetic deterministic analysis contracts, not benchmark observations."""
import copy
import gzip
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from resident_dispatch.analysis import analyze, analyze_pilot, criterion


REGIMES = ("stable", "transient", "persistent", "switching")


def final_fixture(changing=.9, stable=1.0):
    blocks = [f"fresh-{i}" for i in range(8)]
    plan = {"phase": "final", "expected_blocks": blocks, "bootstrap_samples": 200}
    rows = []
    for block, regime, batch, rep, arm in itertools.product(blocks, REGIMES, (128, 1024), (0, 1), ("native", "online")):
        base = 1 + int(block[-1]) / 10 + rep / 100
        ratio = (stable if regime == "stable" else changing) if arm == "online" else 1
        rows.append({"block_id": block, "regime": regime, "batch_tokens": batch,
                     "repetition": rep, "allocation_id": f"allocation-{rep}", "arm": arm,
                     "elapsed_seconds": base * ratio, "status": "complete",
                     "correctness_pass": True, "evidence_eligible": True,
                     "source_map_sha256": "a" * 64,
                     "fixture_only": "Synthetic CPU fixture; not GPU evidence"})
    return plan, rows


def pilot_fixture(candidate=.9, aa=1., adapter=1.):
    blocks = ["pilot-a", "pilot-b"]
    plan = {"expected_blocks": blocks}
    rows = []
    ratios = {"native": 1., "native_aa": aa, "pre_native": adapter,
              "one_copy_home": candidate, "token_balanced": .98}
    for block, domain, batch, rep, arm in itertools.product(blocks, ("domain0", "domain1"), (128, 1024), (0, 1), ratios):
        base = 1. + rep / 100 + (block == "pilot-b") / 10
        rows.append({"block_id": block, "regime": domain, "batch_tokens": batch,
                     "repetition": rep, "allocation_id": "pilot-allocation", "arm": arm,
                     "elapsed_seconds": base * ratios[arm], "status": "complete",
                     "correctness_pass": True, "evidence_eligible": True,
                     "fixture_only": "Synthetic CPU fixture; not GPU evidence"})
    return plan, rows


class ResidentAnalysisTests(unittest.TestCase):
    def test_final_endpoint_complete_matrix_and_allocation_reports(self):
        plan, rows = final_fixture()
        result = analyze(rows, **plan)
        self.assertEqual(result["expected_records"], 256)
        self.assertEqual(result["eligible_matched_pairs"], 128)
        self.assertEqual(result["acceptance"], "pass")
        self.assertTrue(result["claim_eligible"])
        primary = next(x for x in result["aggregates"] if x["group"] == "changing")
        self.assertAlmostEqual(primary["ratio"], .9)
        self.assertEqual(primary["source_blocks"], 8)
        self.assertEqual(len(result["by_allocation"]), 4)
        self.assertEqual({x["family_size"] for x in result["cells"]}, {2, 6})

    def test_useful_gain_and_stable_regression_are_separate_gates(self):
        for changing, stable, useful, guard in ((.96, 1., "fail", "pass"), (.9, 1.03, "pass", "fail")):
            plan, rows = final_fixture(changing, stable)
            result = analyze(rows, **plan)
            self.assertEqual(result["primary_useful_gain"], useful)
            self.assertEqual(result["stable_guard"], guard)
            self.assertEqual(result["acceptance"], "fail")

    def test_exact_interval_boundary_contract(self):
        self.assertEqual(criterion([.9, .95], .95), "pass")
        self.assertEqual(criterion([.95, .96], .95), "inconclusive")
        self.assertEqual(criterion([.950001, .96], .95), "fail")
        self.assertEqual(criterion(None, .95), "inconclusive")

    def test_stable_batch_regression_cannot_hide_in_stable_aggregate(self):
        plan, rows = final_fixture()
        for row in rows:
            if row["arm"] == "online" and row["regime"] == "stable":
                row["elapsed_seconds"] *= 1.03 if row["batch_tokens"] == 128 else .97
        result = analyze(rows, **plan)
        stable = next(x for x in result["aggregates"] if x["group"] == "stable")
        self.assertEqual(stable["criterion"], "pass")
        self.assertEqual(result["stable_guard"], "fail")

    def test_geometric_pairing_and_source_blocks_not_independent_repetitions(self):
        plan, rows = final_fixture(1., 1.)
        for row in rows:
            if row["arm"] == "online":
                row["elapsed_seconds"] *= .64 if row["repetition"] == 0 else 1.25
        result = analyze(rows, **plan)
        for aggregate in result["aggregates"]:
            self.assertAlmostEqual(aggregate["ratio"], math.sqrt(.8))
            self.assertEqual(aggregate["source_blocks"], 8)
        ratios = {round(x["ratio"], 2) for x in result["by_allocation"]}
        self.assertEqual(ratios, {.64, 1.25})

    def test_same_allocation_repetitions_cannot_claim_confirmation(self):
        plan, rows = final_fixture()
        for row in rows:
            row["allocation_id"] = "same-allocation"
        result = analyze(rows, **plan)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["claim_eligible"])
        self.assertEqual(result["acceptance"], "inconclusive")
        self.assertTrue(result["allocation_errors"])

    def test_pair_cannot_cross_allocations(self):
        plan, rows = final_fixture()
        rows[0]["allocation_id"] = "unpaired"
        result = analyze(rows, **plan)
        self.assertEqual(result["acceptance"], "inconclusive")
        self.assertIn("pair crosses allocations", result["invalid_groups"][0]["issues"])

    def test_explicit_allocation_plan_checked(self):
        plan, rows = final_fixture()
        result = analyze(rows, **plan, allocation_by_repetition={"0": "allocation-0", "1": "wrong"})
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(result["allocation_errors"])

    def test_missing_invalid_nonfinite_and_inelegible_rows_fail_closed(self):
        plan, rows = final_fixture()
        self.assertEqual(analyze(rows[1:], **plan)["acceptance"], "inconclusive")
        for key, value in (("elapsed_seconds", float("nan")), ("elapsed_seconds", 0),
                           ("elapsed_seconds", True), ("correctness_pass", False),
                           ("evidence_eligible", False), ("status", "partial")):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            result = analyze(changed, **plan)
            self.assertEqual(result["status"], "incomplete")
            self.assertFalse(result["claim_eligible"])

    def test_duplicates_foreign_arms_and_changed_state_rejected(self):
        plan, rows = final_fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze(rows + [rows[0]], **plan)
        changed = copy.deepcopy(rows)
        changed[0]["arm"] = "posthoc-winner"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            analyze(changed, **plan)
        rows[0]["source_map_sha256"] = "b" * 64
        result = analyze(rows, **plan)
        self.assertIn("paired source_map_sha256 differs", result["invalid_groups"][0]["issues"])

    def test_two_block_validation_has_no_confirmatory_intervals(self):
        plan, rows = final_fixture()
        plan.update(phase="validation", expected_blocks=["fresh-0", "fresh-1"])
        rows = [r for r in rows if r["block_id"] in plan["expected_blocks"]]
        result = analyze(rows, **plan)
        self.assertFalse(result["claim_eligible"])
        self.assertEqual(result["acceptance"], "inconclusive")
        self.assertTrue(all(r["ci95"] is None for r in result["cells"]))

    def test_pilot_null_adapter_and_assignment_are_distinct(self):
        plan, rows = pilot_fixture(candidate=.9, adapter=1.05)
        result = analyze_pilot(rows, **plan)
        self.assertEqual(result["expected_records"], 80)
        self.assertEqual(result["calibration"], "pass")
        self.assertFalse(result["claim_eligible"])
        home, balanced = result["candidates"]
        self.assertTrue(home["advances_to_online_implementation"])
        self.assertFalse(balanced["advances_to_online_implementation"])
        self.assertAlmostEqual(home["candidate_over_pre_native_cells"][0]["ratio"], .9 / 1.05)
        self.assertEqual(result["noise_reserve"], .01)
        self.assertTrue(all("ci95" not in row for row in result["cells"]))

    def test_pilot_requires_noise_plus_selector_tax_headroom(self):
        plan, rows = pilot_fixture(candidate=.94)
        result = analyze_pilot(rows, **plan)
        self.assertEqual(result["calibration"], "pass")
        self.assertFalse(result["candidates"][0]["advances_to_online_implementation"])

    def test_pilot_one_bad_cell_cannot_hide_in_block_aggregate(self):
        plan, rows = pilot_fixture(candidate=.7)
        for row in rows:
            if row["arm"] == "one_copy_home" and (row["block_id"], row["regime"], row["batch_tokens"]) == ("pilot-a", "domain0", 128):
                row["elapsed_seconds"] *= 1.01 / .7
        result = analyze_pilot(rows, **plan)
        self.assertLess(result["candidates"][0]["candidate_over_native_by_block"][0]["ratio"], .93)
        self.assertFalse(result["candidates"][0]["advances_to_online_implementation"])

    def test_bad_aa_stops_even_large_apparent_candidate_gain(self):
        plan, rows = pilot_fixture(candidate=.5, aa=.98)
        result = analyze_pilot(rows, **plan)
        self.assertEqual(result["calibration"], "stop")
        self.assertFalse(any(row["advances_to_online_implementation"] for row in result["candidates"]))

    def test_missing_pilot_cell_is_not_imputed(self):
        plan, rows = pilot_fixture()
        result = analyze_pilot(rows[1:], **plan)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["calibration"], "inconclusive")
        self.assertFalse(any(row["advances_to_online_implementation"] for row in result["candidates"]))

    def test_pilot_candidate_selection_uses_worst_block_and_frozen_tie_rule(self):
        plan, rows = pilot_fixture(candidate=.9)
        for row in rows:
            if row["arm"] == "token_balanced":
                row["elapsed_seconds"] *= .9 / .98
        result = analyze_pilot(rows, **plan)
        self.assertEqual(result["selected_candidate"], "one_copy_home")
        for row in rows:
            if row["arm"] == "token_balanced":
                row["elapsed_seconds"] *= .99
        self.assertEqual(analyze_pilot(rows, **plan)["selected_candidate"], "token_balanced")

    def test_gzip_cli_keeps_pilot_nonconfirmatory(self):
        plan, rows = pilot_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "synthetic.jsonl.gz"
            with gzip.open(records, "wt") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            config = root / "plan.json"
            config.write_text(json.dumps({"phase": "pilot_screen", **plan}))
            output = root / "summary.json"
            completed = subprocess.run([sys.executable, "-m", "resident_dispatch.analysis",
                                        "--plan", str(config), "--records", str(records),
                                        "--output", str(output)], capture_output=True, text=True, check=True)
            result = json.loads(output.read_text())
            self.assertFalse(result["claim_eligible"])
            self.assertEqual(result["expected_records"], 80)
            self.assertEqual(json.loads(completed.stdout)["status"], "complete")


if __name__ == "__main__":
    unittest.main()
