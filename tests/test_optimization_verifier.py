"""Synthetic consistency checks; these fixtures are not GPU observations."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from optimization.analysis import analyze, canonical_hash, manifest_hash
from tests.test_optimization_analysis import fixture, policy_records

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_optimization_results.py"
SPEC = importlib.util.spec_from_file_location("independent_optimization_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def raw_fixture(phase):
    manifest = fixture(phase)
    manifest["policy"] = {"rebalance_interval": 64}
    manifest["default_configs"] = {str(b): {"BLOCK_SIZE_M": 64} for b in (128, 1024)}
    manifest["freeze"]["sha256"] = manifest_hash(manifest)
    rows = policy_records(manifest)
    source = [0, 1]
    digest = hashlib.sha256(struct.pack("<qq", *source)).hexdigest()
    for row in rows:
        if row["kernel_variant"] == "TUNED":
            block = int(row["block_id"][-1])
            row["elapsed_seconds"] *= 1 + block * (.001 if row["regime"] == "stable" else .003)
        row.update(world_size=2, per_rank_elapsed_seconds=[row["elapsed_seconds"]*.99, row["elapsed_seconds"]],
                   kernel_config_rank_agreement=True, source_restore_verified=True, expected_arm_weights_verified=True,
                   commit_completed=True, logical_routing_preserved=True, source_map=source, final_map=source,
                   source_map_sha256=digest, final_map_sha256=digest, migration_bytes=0, accepted=0)
        row["decision_log"] = [
            {"step": step, "allowed": False, "reason": "synthetic fixture only", "current_counts": [128, 128],
             "current_map": source, "proposed_map": source, "committed_map": source, "transfer_sent_bytes": 0,
             "physical_slot_assignments_window": [[64, 64]], "commit_completed": True}
            for step in range(63, 511, 64)]
        row["decision_trajectory_sha256"] = canonical_hash(
            [{f: d.get(f) for f in verifier.TRAJECTORY_FIELDS} for d in row["decision_log"]])
    return manifest, rows, analyze(manifest, rows)


class OptimizationVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.rows, cls.report = raw_fixture("final")

    def verify(self, rows, report=None, manifest=None):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder)/"synthetic.jsonl"
            raw.write_text("".join(json.dumps(row)+"\n" for row in rows))
            return verifier.verify(manifest or self.manifest, [raw], report or self.report)

    def test_complete_eight_block_ratios_intervals_and_counts(self):
        audit = self.verify(self.rows)
        self.assertEqual((audit["records"], audit["matched_four_arm_groups"], audit["source_blocks"]), (512, 128, 8))
        self.assertEqual((audit["verified_cells"], audit["verified_aggregates"]), (48, 18))
        self.assertLessEqual(audit["maximum_absolute_numerical_difference"], 1e-12)
        primary = next(r for r in audit["aggregates"] if r["group"] == "changing" and r["contrast"] == "kernel_B3")
        expected = .9 * math.prod(1 + .003*i for i in range(8))**(1/8)
        self.assertAlmostEqual(primary["ratio"], expected, places=12)
        self.assertEqual(audit["acceptance"], "pass")

    def test_validation_repeats_do_not_create_eight_blocks(self):
        manifest, rows, report = raw_fixture("validation")
        audit = self.verify(rows, report, manifest)
        self.assertEqual(audit["records"], 64)
        self.assertEqual(audit["bootstrap_samples"], 0)
        self.assertTrue(all(row["ci95"] is None for row in audit["aggregates"]))
        self.assertEqual(audit["acceptance"], "inconclusive")

    def test_rejects_missing_arm_and_reported_ratio_change(self):
        with self.assertRaisesRegex(ValueError, "Missing raw arm"):
            self.verify(self.rows[:-1])
        changed = copy.deepcopy(self.report)
        changed["aggregates"][0]["ratio"] += .01
        with self.assertRaisesRegex(ValueError, "Mismatch"):
            self.verify(self.rows, changed)

    def test_rejects_elapsed_inconsistent_with_rank_completion(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["per_rank_elapsed_seconds"][0] = 999.
        with self.assertRaisesRegex(ValueError, "elapsed/max-rank"):
            self.verify(changed)

    def test_rejects_trajectory_change_even_with_recomputed_hash(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["decision_log"][0]["reason"] = "different admission reason"
        changed[0]["decision_trajectory_sha256"] = canonical_hash(
            [{f: d.get(f) for f in verifier.TRAJECTORY_FIELDS} for d in changed[0]["decision_log"]])
        with self.assertRaisesRegex(ValueError, "changed policy trajectory"):
            self.verify(changed)


if __name__ == "__main__":
    unittest.main()
