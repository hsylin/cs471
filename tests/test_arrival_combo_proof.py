"""CPU contracts for the untimed GPU composition proof; no GPU imports."""
import copy
import unittest

import numpy as np

from arrival_combo.proof import (ARMS, check_common_trajectory, check_guard_reports,
                                 routing_arm, validate_arm_evidence)


def fixture():
    mapping = np.concatenate((np.arange(128),np.arange(32)))
    # Shared experts and singleton experts on both owners. Both physical
    # dispatches preserve every logical route, but shared-copy loads differ.
    ids = np.asarray([[0,1,2,3,32,33,80,81], [0,1,2,3,34,35,82,83]],dtype=np.int64)
    logical = np.tile(ids[None,:,:],(48,1,1))
    native = logical.copy()
    native[:,1,:4] += 128
    online = logical.copy()
    counts = {arm:np.stack([np.bincount(row.ravel(),minlength=160) for row in values])
              for arm,values in (('native',native),('online',online))}
    cell = dict(mapping=mapping, logical=logical,counts=counts)
    sequence = list(range(16,48))*4
    proposed = mapping.copy()
    proposed[0],proposed[80] = proposed[80],proposed[0]
    decisions = {}
    for arm in ARMS:
        raw = sequence[48:64]
        decisions[arm] = [dict(step=63,allowed=False,reason='always_skip',
            current_counts=(np.bincount(logical[raw].ravel(),minlength=128)*2).tolist(),
            physical_slot_assignments_window=counts[routing_arm(arm)][raw].tolist(),
            current_map=mapping.tolist(),proposed_map=proposed.tolist(),committed_map=mapping.tolist(),
            transfer_sent_bytes=0,commit_completed=True)]
    return cell,sequence,decisions


def guards(calls=2):
    row = dict(mode='fast',diagnostic_timing=False,pending_pair=False,
        movement_calls=calls,commit_calls=calls,agreement_calls=calls,
        local_unchanged_calls=calls,unanimously_unchanged_calls=calls,
        movement_skips=calls,commit_skips=calls,original_movement_calls=0,
        original_commit_calls=0,fallback_calls=0,diagnostics=[])
    return [row.copy(),row.copy()]


class ArrivalComboProofTests(unittest.TestCase):
    def test_distinct_physical_histories_have_identical_logical_candidate_trajectory(self):
        cell,sequence,decisions = fixture()
        trajectories = {}
        self.assertFalse(np.array_equal(cell['counts']['native'],cell['counts']['online']))
        for arm in ARMS:
            terminal = cell['counts'][routing_arm(arm)][sequence[-16:]]
            trajectories[arm] = validate_arm_evidence(arm,decisions[arm],terminal,cell,sequence)
        self.assertEqual(check_common_trajectory(trajectories),trajectories['native'])

    def test_combined_cannot_claim_native_physical_counters(self):
        cell,sequence,decisions = fixture()
        decisions['combined'][0]['physical_slot_assignments_window'] = decisions['native'][0]['physical_slot_assignments_window']
        with self.assertRaisesRegex(AssertionError,'Arm-specific physical'):
            validate_arm_evidence('combined',decisions['combined'],cell['counts']['online'][sequence[-16:]],cell,sequence)

    def test_exact_logical_vector_and_replicated_rank_multiplicity_required(self):
        for mutation in ('per_rank_only','wrong_expert'):
            cell,sequence,decisions = fixture()
            vector = np.asarray(decisions['combined'][0]['current_counts'])
            if mutation == 'per_rank_only':vector //= 2
            else:vector[0] -= 1;vector[1] += 1
            decisions['combined'][0]['current_counts'] = vector.tolist()
            with self.subTest(mutation=mutation),self.assertRaisesRegex(AssertionError,'exact logical window'):
                validate_arm_evidence('combined',decisions['combined'],cell['counts']['online'][sequence[-16:]],cell,sequence)

    def test_missing_terminal_record_or_extra_terminal_proposal_rejected(self):
        cell,sequence,decisions = fixture()
        with self.assertRaisesRegex(AssertionError,'Terminal arm-specific'):
            validate_arm_evidence('combined',decisions['combined'],np.zeros((16,160)),cell,sequence)
        decisions['combined'].append({**decisions['combined'][0],'step':127})
        with self.assertRaisesRegex(AssertionError,'nonterminal proposal'):
            validate_arm_evidence('combined',decisions['combined'],cell['counts']['online'][sequence[-16:]],cell,sequence)

    def test_no_transfer_and_commit_must_complete(self):
        for field,value in (('transfer_sent_bytes',1),('commit_completed',False),('allowed',True)):
            cell,sequence,decisions = fixture()
            decisions['combined'][0][field] = value
            with self.subTest(field=field),self.assertRaisesRegex(AssertionError,'commit contract'):
                validate_arm_evidence('combined',decisions['combined'],cell['counts']['online'][sequence[-16:]],cell,sequence)

    def test_candidate_changes_are_not_excused_by_dispatch_changes(self):
        cell,sequence,decisions = fixture()
        trajectories = {arm:validate_arm_evidence(arm,decisions[arm],
            cell['counts'][routing_arm(arm)][sequence[-16:]],cell,sequence) for arm in ARMS}
        altered = copy.deepcopy(trajectories)
        altered['combined'][0]['proposed_map'][0] = 17
        with self.assertRaisesRegex(AssertionError,'trajectory differs'):
            check_common_trajectory(altered)

    def test_both_shortcut_arms_include_terminal_pair_and_unanimous_ranks(self):
        check_guard_reports('native',[],2)
        for arm in ('shortcut','combined'):
            check_guard_reports(arm,guards(),2)
            for mutation in ('terminal_missing','pending','one_rank'):
                rows = guards(1) if mutation == 'terminal_missing' else guards()
                if mutation == 'pending':
                    for row in rows:row['pending_pair'] = True
                elif mutation == 'one_rank':rows[1]['commit_skips'] = 1
                with self.subTest(arm=arm,mutation=mutation),self.assertRaises(AssertionError):
                    check_guard_reports(arm,rows,2)

    def test_native_baseline_must_remain_unwrapped(self):
        with self.assertRaisesRegex(AssertionError,'must not be wrapped'):
            check_guard_reports('native',guards(),2)


if __name__ == '__main__':
    unittest.main()
