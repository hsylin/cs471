"""Synthetic CPU contracts for the frozen statistical design; no GPU evidence."""
import copy
import itertools
import math
import unittest

import numpy as np

from arrival_fix.analysis import (ANALYSIS_DEFAULTS, analyze, canonical_hash, infer_endpoints,
                                  passes_upper, stratified_indices, _window_spans)
from arrival_study.schedules import fcfs_metrics
from tests.test_arrival_schedules import row_fixture


def blocks_fixture():
    return {f'arrival-{i}': (f'source-{i//4}', f'domain{(i//2)%2}') for i in range(8)}


def paired_fixture(primary=.9, low=1., tail=1.01, reps=None):
    pairs = []
    for block, batch, pattern, rho, rep in itertools.product(blocks_fixture(), (128, 1024),
            ('spread', 'cluster'), (.5, .85), (0, 1)):
        response = low if rho == .5 else primary
        if reps is not None and rho == .85:
            response = reps[rep]
        pairs.append(dict(arrival_block_id=block, batch_tokens=batch, pattern=pattern, rho=rho,
            repetition=rep, log_ratios=dict(response_mean_ns=math.log(response), response_p95_ns=math.log(tail))))
    return pairs


def full_fixture():
    blocks = [dict(id=key, seed=100+i, block_id=stratum[0], regime=stratum[1])
              for i, (key, stratum) in enumerate(blocks_fixture().items())]
    plan = dict(phase='final', arrival_fix_source_sha256='synthetic-source-only', analysis=ANALYSIS_DEFAULTS.copy(),
        matrix=dict(arrival_blocks=blocks, batches=[128, 1024], patterns=['spread', 'cluster'],
                    rhos=[.5, .85], reps=2, arms=['native', 'fast']), schedules={}, calibration={},
        hardware=dict(hostname='synthetic-host', gpu_uuids=['fixture-gpu0', 'fixture-gpu1']))
    for block, batch in itertools.product(blocks, (128, 1024)):
        plan['calibration'][f"{block['block_id']}/{block['regime']}/B{batch}"] = dict(allocation_id='synthetic-allocation')
        entries = []
        for pattern, rho in itertools.product(('spread', 'cluster'), (.5, .85)):
            gap = 10000 if rho == .5 else 8000
            offsets = [i*gap if pattern == 'spread' else (i//8)*8*gap for i in range(256)]
            entry = dict(pattern=pattern, rho=rho, horizon_ns=256*gap, arrival_offsets_ns=offsets)
            entry['schedule_sha256'] = canonical_hash(entry)
            entries.append(entry)
        plan['schedules'][f"{block['id']}/B{batch}"] = entries
    plan['freeze'] = dict(sha256=canonical_hash(plan))
    rows = []
    for block, batch, pattern, rho, rep in itertools.product(blocks, (128, 1024), ('spread', 'cluster'), (.5, .85), (0, 1)):
        schedule = next(x for x in plan['schedules'][f"{block['id']}/B{batch}"] if x['pattern']==pattern and x['rho']==rho)
        order = ['native', 'fast'] if rep == 0 else ['fast', 'native']
        for arm in order:
            row = row_fixture(schedule['arrival_offsets_ns'][:], [1000 if arm == 'native' else 900]*256,
                              horizon=schedule['horizon_ns'])
            row.update(arrival_block_id=block['id'], arrival_seed=block['seed'], block_id=block['block_id'], regime=block['regime'],
                batch_tokens=batch, pattern=pattern, rho=rho, repetition=rep, arm=arm, arm_order=order[:],
                phase='final', status='complete', config_hash=plan['freeze']['sha256'],
                arrival_fix_source_sha256='synthetic-source-only', allocation_id='synthetic-allocation',
                evidence_eligible=True, diagnostic_only=False, guard_symbols_restored=True,
                hostname='synthetic-host', physical_gpu_uuids=['fixture-gpu0', 'fixture-gpu1'],
                arrival_schedule=copy.deepcopy(schedule), correctness_pass=True, preflight_correctness={'correctness_pass': True},
                normal_completion_verified=True, resident_state_verified=True, rank_assignment_agreement=True,
                migration_bytes=0, migration_sent_messages=0, terminal_proposals_suppressed=1, policy='SKIP',
                source_map=[0, 1], final_map=[0, 1], capture_sha256='capture', replay_sha256='replay',
                source_prompt_ids=['doc-a', 'doc-b'], source_weights_sha256='weights', source_weight_rank_sha256=['r0', 'r1'],
                resolved_kernel_config={}, logical_routing_sha256='logical', decision_trajectory_sha256='trajectory',
                arrival_clock_contract=dict(implementation='clock_gettime(CLOCK_MONOTONIC)', monotonic=True, adjustable=False),
                warmup_raw_indices=list(range(16)), conditioning_raw_indices=list(range(64)),
                measured_raw_indices=list(range(256)), assignment_sha256='same-native-routing')
            guard = dict(mode='fast', diagnostic_timing=False, pending_pair=False,
                movement_calls=4, commit_calls=4, agreement_calls=4, local_unchanged_calls=4,
                unanimously_unchanged_calls=4, movement_skips=4, commit_skips=4,
                original_movement_calls=0, original_commit_calls=0, fallback_calls=0, diagnostics=[])
            row['noop_guard_ranks'] = [guard.copy(), guard.copy()] if arm == 'fast' else []
            row['queue_metrics'] = fcfs_metrics(row)
            row['observed_window_spans'] = _window_spans(row)
            rows.append(row)
    return plan, rows


class ArrivalFixInferenceTests(unittest.TestCase):
    def test_primary_strict_boundary_and_guard_inclusive_boundary(self):
        self.assertFalse(passes_upper(.95, .95, strict=True))
        self.assertTrue(passes_upper(.9499999, .95, strict=True))
        self.assertTrue(passes_upper(1.02, 1.02))
        self.assertFalse(passes_upper(1.0200001, 1.02))

    def test_stratified_draws_preserve_two_blocks_per_fixed_payload_state(self):
        strata = blocks_fixture()
        blocks, draws = stratified_indices(strata, samples=200, seed=471420)
        self.assertTrue(np.array_equal(draws, stratified_indices(strata, samples=200, seed=471420)[1]))
        for row in draws:
            selected = [strata[blocks[i]] for i in row]
            self.assertEqual(sorted(selected), sorted(strata.values()))

    def test_fixed_stratum_means_do_not_gain_artificial_composition_uncertainty(self):
        pairs = paired_fixture(primary=1., low=1., tail=1.)
        stratum_logs = {key: value for key, value in zip(sorted(set(blocks_fixture().values())), (-.2, -.1, .1, .2))}
        for pair in pairs:
            if pair['rho'] == .85:
                pair['log_ratios']['response_mean_ns'] = stratum_logs[blocks_fixture()[pair['arrival_block_id']]]
        result = infer_endpoints(pairs, blocks_fixture())
        np.testing.assert_allclose(result['primary']['ci95'], [1., 1.], rtol=0, atol=1e-15)

    def test_repeat_and_cell_aggregation_uses_paired_logs(self):
        result = infer_endpoints(paired_fixture(reps=(.8, 1.)), blocks_fixture())
        self.assertAlmostEqual(result['primary']['ratio'], math.sqrt(.8))
        self.assertEqual(result['acceptance'], 'pass')
        self.assertEqual(result['bootstrap']['bootstrap_samples'], 100000)
        self.assertEqual(result['guards'][0]['one_sided_confidence_level'], .975)
        self.assertEqual(result['guards'][1]['guard_family_size'], 2)

    def test_guard_failure_prevents_acceptance_without_claiming_proven_regression(self):
        result = infer_endpoints(paired_fixture(tail=1.03), blocks_fixture())
        self.assertTrue(result['primary']['passes'])
        self.assertEqual(result['guards'][1]['status'], 'not_established')
        self.assertEqual(result['acceptance'], 'not_established')

    def test_missing_repeat_duplicate_or_changed_analysis_cannot_silently_change_units(self):
        for mutation in ('missing', 'duplicate', 'options', 'unbalanced_strata'):
            pairs, strata = paired_fixture(), blocks_fixture()
            options = {}
            if mutation == 'missing': pairs.pop()
            elif mutation == 'duplicate': pairs.append(pairs[0])
            elif mutation == 'options': options['bootstrap_samples'] = 99999
            else: strata['arrival-0'] = strata['arrival-2']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                infer_endpoints(pairs, strata, **options)


class ArrivalFixRawAnalysisTests(unittest.TestCase):
    def test_complete_matrix_and_raw_clock_validation_produce_only_declared_inference(self):
        plan, rows = full_fixture()
        result = analyze(plan, rows)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['observed_records'], 256)
        self.assertEqual(result['matched_pairs'], 128)
        self.assertEqual(len(result['descriptive_cells']), 8)
        self.assertEqual(result['confirmatory']['acceptance'], 'pass')
        self.assertAlmostEqual(result['confirmatory']['primary']['ratio'], .9)
        self.assertNotIn('ci95', result['descriptive_cells'][0])

    def test_incomplete_matrix_or_stored_clock_tampering_disables_all_confirmatory_output(self):
        for mutation in ('missing', 'summary', 'schedule', 'order', 'source', 'pair_input', 'hardware', 'guard', 'eligibility'):
            plan, rows = full_fixture()
            if mutation == 'missing': rows.pop()
            elif mutation == 'summary': rows[0]['queue_metrics']['summary_ns']['response']['mean'] += 1
            elif mutation == 'schedule': rows[0]['arrival_offsets_ns'][1] += 1
            elif mutation == 'order': rows[0], rows[1] = rows[1], rows[0]
            elif mutation == 'source': rows[0]['arrival_fix_source_sha256'] = 'changed'
            elif mutation == 'hardware': rows[0]['physical_gpu_uuids'][0] = 'wrong-gpu'
            elif mutation == 'guard': rows[1]['noop_guard_ranks'][0]['commit_skips'] = 3
            elif mutation == 'eligibility': rows[0]['evidence_eligible'] = False
            else: rows[0]['capture_sha256'] = 'different-input'
            with self.subTest(mutation=mutation):
                result = analyze(plan, rows)
                self.assertFalse(result['valid'])
                self.assertIsNone(result['confirmatory'])

    def test_pilot_and_calibration_cannot_enter_final_analysis(self):
        plan, rows = full_fixture()
        for phase in ('pilot', 'calibration'):
            plan['phase'] = phase
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                analyze(plan, rows)


if __name__ == '__main__':
    unittest.main()
