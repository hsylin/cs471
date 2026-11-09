"""Synthetic inference/clock contracts, never GPU experimental observations."""
import copy
import itertools
import math
import unittest

from arrival_combo.analysis import (ANALYSIS_DEFAULTS, ARMS, CONTRASTS, analyze,
    canonical_hash, infer_endpoints, _order_key, _window_spans)
from arrival_study.schedules import fcfs_metrics
from tests.test_arrival_fix_analysis import blocks_fixture, full_fixture as prior_fixture
from tests.test_arrival_schedules import row_fixture


def pair_fixture(increment=.9, baseline=.7, small_batch=None, low=1., p95=1.):
    result=[]
    for block,batch,pattern,rho,rep in itertools.product(blocks_fixture(),(128,1024),('spread','cluster'),(.5,.85),(0,1)):
        ratio=low if rho==.5 else increment
        if small_batch is not None and batch==128 and rho==.85:ratio=small_batch
        logs={'combined_over_shortcut':math.log(ratio),'combined_over_native':math.log(baseline),
              'shortcut_over_native':math.log(baseline)-math.log(ratio)}
        for contrast in CONTRASTS:
            result.append(dict(arrival_block_id=block,batch_tokens=batch,pattern=pattern,rho=rho,repetition=rep,
                contrast=contrast,log_ratios=dict(response_mean_ns=logs[contrast],
                    response_p95_ns=math.log(p95) if contrast=='combined_over_shortcut' else math.log(baseline))))
    return result


def full_fixture():
    plan,old_rows=prior_fixture()
    plan['matrix']['arms']=ARMS[:]
    plan['analysis']=ANALYSIS_DEFAULTS.copy()
    plan['arrival_combo_source_sha256']=plan.pop('arrival_fix_source_sha256')
    plan['arrival_combo_schema_version']=1
    groups={tuple(row[key] for key in ('arrival_block_id','batch_tokens','pattern','rho','repetition')):row
            for row in old_rows if row['arm']=='native'}
    plan['arm_orders']={}
    for i,key in enumerate(sorted({key[:4] for key in groups})):
        shift=i%3;base=ARMS[shift:]+ARMS[:shift]
        plan['arm_orders'][_order_key(*key)]=[base,list(reversed(base))]
    plan['freeze']['sha256']=canonical_hash({k:v for k,v in plan.items() if k!='freeze'})
    rows=[]
    for key,source in groups.items():
        order=plan['arm_orders'][_order_key(*key[:4])][key[4]]
        for arm in order:
            row=copy.deepcopy(source)
            row.update(row_fixture(row['arrival_offsets_ns'],[{'native':1000,'shortcut':900,'combined':840}[arm]]*256,
                                   horizon=row['horizon_ns']))
            row.update(arm=arm,arm_order=order[:],config_hash=plan['freeze']['sha256'],arrival_combo_schema_version=1,
                arrival_combo_source_sha256=plan['arrival_combo_source_sha256'],
                dispatch_mode='online' if arm=='combined' else 'native',noop_enabled=arm!='native',
                assignment_selection_cost_included=True,online_residency_metadata_verified=arm=='combined',
                mapper_calls_measured=256 if arm=='combined' else 0,rank_gpu_uuids=plan['hardware']['gpu_uuids'][:])
            row.pop('arrival_fix_source_sha256')
            row['assignment_sha256']='online-route' if arm=='combined' else 'native-route'
            guard=dict(mode='fast',diagnostic_timing=False,pending_pair=False,movement_calls=4,commit_calls=4,
                agreement_calls=4,local_unchanged_calls=4,unanimously_unchanged_calls=4,movement_skips=4,commit_skips=4,
                original_movement_calls=0,original_commit_calls=0,fallback_calls=0,diagnostics=[])
            row['noop_guard_ranks']=[] if arm=='native' else [guard.copy(),guard.copy()]
            row['queue_metrics']=fcfs_metrics(row)
            row['observed_window_spans']=_window_spans(row)
            rows.append(row)
    return plan,rows


class ArrivalComboStatisticalTests(unittest.TestCase):
    def test_baseline_gain_cannot_rescue_failed_incremental_primary(self):
        result=infer_endpoints(pair_fixture(increment=.99,baseline=.5),blocks_fixture())
        self.assertEqual(result['incremental_acceptance'],'not_established')
        secondary=result['secondary_baseline']
        self.assertAlmostEqual(secondary['ratio'],.5)
        self.assertEqual(secondary['status'],'not_tested_gate_closed')
        self.assertIsNone(secondary['ci95'])
        self.assertIsNone(secondary['passes'])
        self.assertEqual(result['acceptance'],'not_established')

    def test_small_batch_guard_prevents_large_batch_masking(self):
        result=infer_endpoints(pair_fixture(increment=.8,small_batch=1.03),blocks_fixture())
        self.assertTrue(result['primary']['passes'])
        small=next(g for g in result['guards'] if g['name']=='b128_high_load_mean_response')
        self.assertFalse(small['passes'])
        self.assertEqual(result['secondary_baseline']['status'],'not_tested_gate_closed')

    def test_three_guard_bounds_and_hierarchical_secondary(self):
        result=infer_endpoints(pair_fixture(),blocks_fixture())
        self.assertEqual(result['acceptance'],'pass')
        self.assertTrue(result['secondary_baseline']['confirmatory'])
        for guard in result['guards']:
            self.assertEqual(guard['guard_family_size'],3)
            self.assertAlmostEqual(guard['one_sided_confidence_level'],1-.05/3)
        self.assertEqual(result['bootstrap']['bootstrap_seed'],471432)
        self.assertEqual(result['bootstrap']['bootstrap_samples'],100000)

    def test_secondary_has_its_own_requirement_after_open_gate(self):
        result=infer_endpoints(pair_fixture(baseline=1.01),blocks_fixture())
        self.assertEqual(result['incremental_acceptance'],'pass')
        self.assertFalse(result['secondary_baseline']['passes'])
        self.assertEqual(result['acceptance'],'not_established')

    def test_incomplete_contrast_matrix_cannot_increase_or_change_units(self):
        pairs=pair_fixture()
        for changed in (pairs[:-1],pairs+[pairs[0]]):
            with self.assertRaises(ValueError):infer_endpoints(changed,blocks_fixture())


class ArrivalComboRawTests(unittest.TestCase):
    def test_three_arms_keep_logical_identity_and_allow_combined_physical_changes(self):
        plan,rows=full_fixture();result=analyze(plan,rows)
        self.assertTrue(result['valid'],result['errors'])
        self.assertEqual(result['expected_records'],384)
        self.assertEqual(result['matched_triplets'],128)
        self.assertEqual(result['matched_pairs'],384)
        self.assertEqual(set(result['pairs_per_contrast'].values()),{128})
        self.assertEqual(len(result['descriptive_cells']),8)
        self.assertEqual(set(result['descriptive_cells'][0]['contrasts']),set(CONTRASTS))
        self.assertEqual(result['confirmatory']['acceptance'],'pass')
        self.assertAlmostEqual(result['confirmatory']['primary']['ratio'],.84/.9)
        for counts in result['arm_position_counts'].values():
            self.assertEqual(sum(counts),128)

    def test_wrong_native_assignment_logical_route_mapper_or_order_disables_inference(self):
        for mutation in ('native_assignment','logical','mapper','order','missing','source','guard'):
            plan,rows=full_fixture()
            shortcut=next(row for row in rows if row['arm']=='shortcut')
            combined=next(row for row in rows if row['arm']=='combined')
            if mutation=='native_assignment':shortcut['assignment_sha256']='wrong-native-assignment'
            elif mutation=='logical':combined['logical_routing_sha256']='changed-logical-routing'
            elif mutation=='mapper':combined['mapper_calls_measured']=255
            elif mutation=='order':rows[0],rows[1]=rows[1],rows[0]
            elif mutation=='missing':rows.pop()
            elif mutation=='source':combined['arrival_combo_source_sha256']='changed'
            else:shortcut['noop_guard_ranks'][0]['commit_skips']=3
            with self.subTest(mutation=mutation):
                result=analyze(plan,rows)
                self.assertFalse(result['valid'])
                self.assertIsNone(result['confirmatory'])


if __name__=='__main__':unittest.main()
