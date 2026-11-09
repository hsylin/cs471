"""Synthetic CPU contracts for the independent final statistics verifier."""
import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("resident_final_verifier", ROOT / "scripts/verify_resident_final.py")
verify_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_module)


def fixture(changing=.9, stable=(1., 1.)):
    m = json.loads((ROOT / "configs/resident-online-final-frozen-v1.json").read_text())
    x = m["matrix"]
    blocks, regimes, batches = x["block_ids"], x["regimes"], x["batch_tokens"]
    tapes = {(t["block_id"], t["regime"], t["batch_tokens"]): t for t in m["tapes"]}
    docs = {b["block_id"]: b["source_document_ids"] for b in m["blocks"]}
    cells = list(itertools.product(blocks, regimes, batches))
    random.Random(m["execution"]["order_seed"]).shuffle(cells)
    mapping = list(range(128)) + list(range(32))
    rows, pairs = [], []
    for rep in (0, 1):
        for position, (block, regime, batch) in enumerate(cells):
            order = ["native", "online"] if position % 2 == 0 else ["online", "native"]
            if rep:
                order.reverse()
            ratio = stable[batches.index(batch)] if regime == "stable" else changing
            base = 1 + blocks.index(block) / 10 + rep / 100
            counts = [16 * batch * 2] * 8 + [0] * 120
            decisions = [{"step": step, "current_counts": counts, "current_map": mapping,
                          "proposed_map": mapping, "committed_map": mapping, "allowed": False}
                         for step in range(63, 511, 64)]
            trajectory = verify_module.digest([{k: d[k] for k in ("step", "current_counts", "current_map", "proposed_map", "committed_map")} for d in decisions])
            for arm in order:
                elapsed = base * (ratio if arm == "online" else 1.)
                rows.append({"block_id": block, "regime": regime, "batch_tokens": batch,
                    "repetition": rep, "arm": arm, "allocation_id": f"synthetic-{rep}",
                    "config_hash": m["freeze"]["sha256"], "phase": "final", "split": "final",
                    "status": "complete", "policy": "SKIP", "steps_completed": 512,
                    "elapsed_seconds": elapsed, "per_rank_elapsed_seconds": [elapsed * .99, elapsed],
                    "resident_dispatch_sha256": m["environment"]["resident_dispatch_sha256"],
                    "resident_dispatch_file_sha256": m["execution_source_sha256"],
                    "source_map": mapping, "final_map": mapping, "source_map_sha256": "a" * 64,
                    "source_weights_sha256": "b" * 64, "capture_sha256": tapes[block, regime, batch]["capture_sha256"],
                    "replay_sha256": tapes[block, regime, batch]["replay_sha256"],
                    "logical_routing_sha256": "c" * 64, "decision_log": decisions,
                    "decision_trajectory_sha256": trajectory, "logical_trajectory_sha256": trajectory,
                    "source_prompt_ids": docs[block], "resolved_kernel_config": m["default_configs"][str(batch)],
                    "timing_scope": m["execution"]["timing_scope"], "arm_order": order.copy(), "arm_position": order.index(arm),
                    "online_algorithm": m["online_algorithm"] if arm == "online" else "native_knuth_hash",
                    "migration_bytes": 0, "migration_sent_messages": 0, "terminal_proposals_suppressed": 1,
                    "measurement_setup_revision": 2, "full_policy_conditioning_steps": 64,
                    "full_policy_conditioning_calls": 1,
                    **{flag: True for flag in ("correctness_pass", "evidence_eligible", "assignment_selection_cost_included",
                        "source_restore_verified", "expected_arm_weights_verified", "commit_completed", "logical_routing_preserved",
                        "rank_assignment_agreement", "timing_events_initialized_before_timer", "gc_enabled_during_measurement")},
                    "fixture_only": "Synthetic CPU fixture; not GPU benchmark evidence"})
            pairs.append({"block_id": block, "regime": regime, "batch_tokens": batch,
                          "repetition": rep, "allocation_id": f"synthetic-{rep}",
                          "native_seconds": base, "treatment_seconds": base * ratio, "log_ratio": math.log(ratio)})
    cell_summary = []
    for regime, batch in itertools.product(regimes, batches):
        ratio = stable[batches.index(batch)] if regime == "stable" else changing
        limit, family = (1.02, 2) if regime == "stable" else (.95, 6)
        cell_summary.append({"regime": regime, "batch_tokens": batch, "source_blocks": 8,
                             "ratio": ratio, "ci95": [ratio, ratio], "adjusted_ci": [ratio, ratio],
                             "family_size": family, "limit": limit,
                             "criterion": "pass" if ratio < limit else "fail"})
    stable_ratio = math.sqrt(stable[0] * stable[1])
    useful = "pass" if changing < .95 else "fail"
    guard = "pass" if max(stable) < 1.02 else "fail"
    aggregates = []
    for group, ratio, limit in (("changing", changing, .95), ("stable", stable_ratio, 1.02)):
        aggregates.append({"group": group, "source_blocks": 8, "ratio": ratio, "ci95": [ratio, ratio],
                           "adjusted_ci": [ratio, ratio], "family_size": 1, "limit": limit,
                           "criterion": "pass" if ratio < limit else "fail"})
    summary = {"phase": "final", "status": "complete", "expected_records": 256, "observed_records": 256,
               "eligible_matched_pairs": 128, "claim_eligible": True, "native_arm": "native", "treatment_arm": "online",
               "primary_useful_gain": useful, "stable_guard": guard,
               "acceptance": "pass" if useful == guard == "pass" else "fail",
               "missing": [], "invalid_groups": [], "allocation_errors": [],
               "allocations_by_repetition": {"0": ["synthetic-0"], "1": ["synthetic-1"]},
               "pairs": pairs, "cells": cell_summary, "aggregates": aggregates,
               "by_allocation": [{"allocation_id": f"synthetic-{rep}", "group": group,
                                   "pairs": 16 if group == "stable" else 48,
                                   "ratio": stable_ratio if group == "stable" else changing}
                                  for rep, group in itertools.product((0, 1), ("changing", "stable"))]}
    return m, rows, summary


class ResidentFinalVerifierTests(unittest.TestCase):
    def test_complete_known_effect_matrix_and_eight_block_units(self):
        result = verify_module.verify(*fixture())
        self.assertTrue(result["pass"], result["errors"][:3])
        self.assertEqual(result["acceptance"], "pass")
        self.assertEqual(result["source_blocks"], 8)
        self.assertEqual(result["matched_pairs"], 128)
        self.assertLess(result["maximum_numeric_difference"], 1e-12)

    def test_fixed_gain_fail_and_individual_stable_guard_fail(self):
        for arguments, useful, guard in (({"changing": .96}, "fail", "pass"),
                                         ({"stable": (1.03, .97)}, "pass", "fail")):
            result = verify_module.verify(*fixture(**arguments))
            self.assertTrue(result["pass"], result["errors"][:3])
            self.assertEqual(result["primary_useful_gain"], useful)
            self.assertEqual(result["stable_guard"], guard)
            self.assertEqual(result["acceptance"], "fail")

    def test_exact_threshold_and_nonconstant_block_bootstrap(self):
        self.assertEqual(verify_module.decision([.94, .95], .95), "pass")
        self.assertEqual(verify_module.decision([.95, .96], .95), "inconclusive")
        self.assertEqual(verify_module.decision([.951, .96], .95), "fail")
        logs = np.log([.8] + [1.] * 7)
        estimate = verify_module.estimate(logs, verify_module.draw_blocks(), family=6)
        self.assertAlmostEqual(estimate["ratio"], .8 ** (1 / 8))
        self.assertAlmostEqual(estimate["ci95"][0], .8 ** (3 / 8))
        self.assertEqual(estimate["ci95"][1], 1.)
        self.assertAlmostEqual(estimate["adjusted_ci"][0], .8 ** (4 / 8))
        with self.assertRaises(ValueError):
            verify_module.estimate(np.tile(logs, 2), verify_module.draw_blocks())

    def test_missing_duplicate_or_shared_allocation_fail_before_statistics(self):
        m, rows, summary = fixture()
        for bad in (rows[:-1], rows + [rows[0]]):
            result = verify_module.verify(m, bad, summary)
            self.assertFalse(result["pass"])
            self.assertFalse(result["statistics_recomputed"])
        bad = copy.deepcopy(rows)
        for row in bad:
            row["allocation_id"] = "same-job"
        result = verify_module.verify(m, bad, summary)
        self.assertFalse(result["pass"])
        self.assertFalse(result["statistics_recomputed"])

    def test_authorized_allocation_and_pair_state_changes_checked(self):
        m, rows, summary = fixture()
        result = verify_module.verify(m, rows, summary, allocation_by_repetition={0: "wrong", 1: "synthetic-1"})
        self.assertFalse(result["pass"])
        rows[0]["source_weights_sha256"] = "e" * 64
        result = verify_module.verify(m, rows, summary)
        self.assertFalse(result["pass"])
        self.assertFalse(result["statistics_recomputed"])

    def test_wrong_summary_interval_or_outcome_is_detected(self):
        m, rows, summary = fixture()
        summary["cells"][0]["adjusted_ci"][1] += .01
        result = verify_module.verify(m, rows, summary)
        self.assertFalse(result["pass"])
        self.assertTrue(result["statistics_recomputed"])
        m, rows, summary = fixture()
        summary["acceptance"] = "fail"
        self.assertFalse(verify_module.verify(m, rows, summary)["pass"])


if __name__ == "__main__":
    unittest.main()
