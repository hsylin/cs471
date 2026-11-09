"""Synthetic CPU fixtures for the independent pilot verifier, not GPU evidence."""
import copy
import gzip
import importlib.util
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_resident_headroom.py"
spec = importlib.util.spec_from_file_location("independent_resident_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def fixture(aa=1., home=.9, balanced=.98, expected_selection="one_copy_home"):
    root = SCRIPT.parent.parent
    manifest = json.loads((root / "configs/resident-headroom-frozen-v1.json").read_text())
    matrix = manifest["matrix"]
    blocks, batches = matrix["block_ids"], matrix["batch_tokens"]
    regimes = [f"domain{x}" for x in matrix["domains"]]
    banks = {x["block_id"]: x["capture_sha256"] for x in manifest["banks"]}
    docs = {x["block_id"]: x["source_document_ids"] for x in manifest["blocks"]}
    ratios = {"native": 1., "native_aa": aa, "pre_native": 1.03,
              "one_copy_home": home, "token_balanced": balanced}
    mapping = list(range(128)) + list(range(32))
    cells = list(itertools.product(blocks, matrix["domains"], batches))
    random.Random(manifest["execution"]["order_seed"]).shuffle(cells)
    rows = []
    for index, (block, domain, batch) in enumerate(cells):
        shift = index % 5
        first = verifier.ARMS[shift:] + verifier.ARMS[:shift]
        counts = [16 * batch * 2] * 8 + [0] * 120
        decisions = [{"step": step, "current_counts": counts, "current_map": mapping,
                      "proposed_map": mapping, "committed_map": mapping, "allowed": False}
                     for step in (63, 127, 191)]
        trajectory = [{key: d[key] for key in ("step", "current_counts", "current_map", "proposed_map", "committed_map")} for d in decisions]
        for rep, order in enumerate((first, first[::-1])):
            for pos, arm in enumerate(order):
                elapsed = (1 + index / 10 + rep / 100) * ratios[arm]
                rows.append({"block_id": block, "regime": f"domain{domain}", "batch_tokens": batch,
                    "repetition": rep, "arm": arm, "allocation_id": "synthetic-test-only",
                    "elapsed_seconds": elapsed, "per_rank_elapsed_seconds": [elapsed * .99, elapsed],
                    "status": "complete", "correctness_pass": True, "evidence_eligible": True,
                    "source_restore_verified": True, "expected_arm_weights_verified": True,
                    "commit_completed": True, "logical_routing_preserved": True,
                    "rank_assignment_agreement": True, "config_hash": manifest["freeze"]["sha256"],
                    "resident_dispatch_sha256": manifest["environment"]["resident_dispatch_sha256"],
                    "resident_dispatch_file_sha256": manifest["execution_source_sha256"],
                    "capture_sha256": banks[block], "source_prompt_ids": docs[block],
                    "source_map_sha256": "a" * 64, "source_weights_sha256": "b" * 64,
                    "source_map": mapping, "final_map": mapping, "decision_log": decisions,
                    "logical_trajectory_sha256": verifier.digest(trajectory),
                    "resolved_kernel_config": manifest["default_configs"][str(batch)],
                    "timing_scope": manifest["execution"]["timing_scope"], "steps_completed": 256,
                    "terminal_proposals_suppressed": 1, "migration_bytes": 0,
                    "migration_sent_messages": 0, "policy": "SKIP", "arm_order": list(order),
                    "arm_position": pos, "assignment_sha256": ("c" if arm in verifier.ARMS[:3] else "d") * 64,
                    "fixture_only": "Synthetic CPU fixture; not GPU benchmark evidence"})
    calibrated = .99 <= aa <= 1.01
    summary = {"status": "complete", "expected_records": 80, "observed_records": 80,
               "claim_eligible": False, "calibration": "pass" if calibrated else "stop",
               "null_aggregate_ratio": aa, "max_null_cell_deviation": abs(aa - 1),
               "noise_reserve": max(.01, abs(aa - 1)), "selector_tax_reserve": .01,
               "selected_candidate": expected_selection, "missing": [], "invalid_groups": [], "allocation_errors": [],
               "cells": [{"block_id": block, "regime": regime, "batch_tokens": batch, "arm": arm, "ratio": ratios[arm]}
                         for block, regime, batch, arm in itertools.product(blocks, regimes, batches, verifier.ARMS[1:])],
               "candidates": []}
    for arm in verifier.CANDIDATES:
        passes = calibrated and ratios[arm] < .93
        summary["candidates"].append({"arm": arm, "advances_to_online_implementation": passes,
            "screen": "pass" if passes else "stop",
            "candidate_over_native_by_block": [{"block_id": block, "ratio": ratios[arm]} for block in blocks],
            "candidate_over_pre_native_cells": [{"block_id": block, "regime": regime, "batch_tokens": batch, "ratio": ratios[arm] / 1.03}
                for block, regime, batch in itertools.product(blocks, regimes, batches)]})
    return manifest, rows, summary


class IndependentResidentVerifierTests(unittest.TestCase):
    def test_independent_product_root_statistics_match_known_fixture(self):
        result = verifier.verify(*fixture())
        self.assertTrue(result["pass"], result["errors"])
        self.assertEqual(result["selected_candidate"], "one_copy_home")
        self.assertEqual(result["matched_five_arm_repetitions"], 16)
        self.assertLess(result["maximum_cell_ratio_difference"], 1e-12)

    def test_aa_failure_is_valid_audit_but_stops_selection(self):
        result = verifier.verify(*fixture(aa=.98, expected_selection=None))
        self.assertTrue(result["pass"], result["errors"])
        self.assertEqual(result["calibration"], "stop")
        self.assertIsNone(result["selected_candidate"])
        self.assertFalse(any(c["advances"] for c in result["candidates"]))

    def test_thresholds_charge_both_reserves_and_cell_guard(self):
        self.assertTrue(verifier.advance([.929, .92], [.999, .9], True, .01, .01))
        self.assertFalse(verifier.advance([.931, .92], [.9], True, .01, .01))
        self.assertFalse(verifier.advance([.9], [1.001], True, .01, .01))
        self.assertFalse(verifier.advance([.5], [.5], False, .01, .01))

    def test_exact_mechanism_tie_chooses_home(self):
        result = verifier.verify(*fixture(home=.9, balanced=.9))
        self.assertTrue(result["pass"], result["errors"])
        self.assertEqual(result["selected_candidate"], "one_copy_home")

    def test_missing_duplicate_and_foreign_rows_are_not_imputed(self):
        manifest, rows, summary = fixture()
        for bad in (rows[:-1], rows + [rows[0]]):
            result = verifier.verify(manifest, bad, summary)
            self.assertFalse(result["pass"])
            self.assertFalse(result["selection_recomputed"])
        bad = copy.deepcopy(rows)
        bad[0]["arm"] = "unplanned"
        self.assertFalse(verifier.verify(manifest, bad, summary)["pass"])

    def test_invalidated_attempt_and_mixed_allocations_are_rejected(self):
        manifest, rows, summary = fixture()
        rejected = verifier.verify(manifest, rows, summary, expected_allocation_id="different-clean-attempt")
        self.assertFalse(rejected["pass"])
        self.assertFalse(rejected["selection_recomputed"])
        rows[0]["allocation_id"] = "invalidated-attempt"
        self.assertFalse(verifier.verify(manifest, rows, summary)["pass"])

    def test_state_cadence_slowest_rank_and_order_are_checked(self):
        manifest, rows, summary = fixture()
        for field, value in (("source_weights_sha256", "f" * 64),
                             ("elapsed_seconds", rows[0]["elapsed_seconds"] * 1.1),
                             ("decision_log", []), ("arm_position", 99),
                             ("correctness_pass", False), ("final_map", [0] * 160)):
            bad = copy.deepcopy(rows)
            bad[0][field] = value
            result = verifier.verify(manifest, bad, summary)
            self.assertFalse(result["pass"], field)
            self.assertFalse(result["selection_recomputed"], field)

    def test_wrong_analyzer_ratio_gate_or_selection_detected(self):
        manifest, rows, summary = fixture()
        for field, value in (("calibration", "stop"), ("selected_candidate", "token_balanced"),
                             ("noise_reserve", 0), ("claim_eligible", True)):
            bad = copy.deepcopy(summary)
            bad[field] = value
            self.assertFalse(verifier.verify(manifest, rows, bad)["pass"], field)
        summary["cells"][0]["ratio"] += .01
        self.assertFalse(verifier.verify(manifest, rows, summary)["pass"])

    def test_portable_gzip_cli(self):
        manifest, rows, summary = fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path, summary_path = root / "manifest.json", root / "summary.json"
            records, output = root / "synthetic.jsonl.gz", root / "audit.json"
            manifest_path.write_text(json.dumps(manifest))
            summary_path.write_text(json.dumps(summary))
            with gzip.open(records, "wt") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest_path),
                            "--records", str(records), "--summary", str(summary_path),
                            "--output", str(output)], capture_output=True, text=True, check=True)
            self.assertTrue(json.loads(output.read_text())["pass"])


if __name__ == "__main__":
    unittest.main()
