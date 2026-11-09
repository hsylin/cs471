"""Independent synthetic contracts; no production analysis import or GPU data."""
import copy
import importlib.util
import itertools
import math
from pathlib import Path
import unittest

import numpy as np

from tests.test_verify_arrival_fix_statistics import full_fixture as previous_fixture, row_clocks, strata_fixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('independent_combo_statistics',
    ROOT/'scripts/verify_arrival_combo_statistics.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def triplets_fixture(increment=lambda block,batch,pattern,rho,rep,metric:.9, baseline=.6):
    result = []
    for block,batch,pattern,rho,rep in itertools.product(strata_fixture(),(128,1024),('spread','cluster'),(.5,.85),(0,1)):
        logs = {name:{} for name in audit.CONTRASTS}
        for metric in audit.METRICS:
            inc = increment(block,batch,pattern,rho,rep,metric)
            logs['combined_over_shortcut'][metric] = math.log(inc)
            logs['combined_over_native'][metric] = math.log(baseline)
            logs['shortcut_over_native'][metric] = math.log(baseline)-math.log(inc)
        result.append(dict(arrival_block_id=block,batch_tokens=batch,pattern=pattern,rho=rho,
            repetition=rep,log_ratios=logs))
    return result


def full_fixture():
    plan,old_rows = previous_fixture()
    plan.pop('arrival_fix_source_sha256')
    plan['arrival_combo_source_sha256'] = 'synthetic-combo-only'
    plan['matrix']['arms'] = list(audit.ARMS)
    plan['analysis'] = audit.SETTINGS.copy()
    plan['arm_orders'] = {}
    old_index = {tuple(row[k] for k in audit.KEY):row for row in old_rows}
    rows = []
    units = sorted({tuple(row[k] for k in audit.UNIT) for row in old_rows})
    for i,key in enumerate(units):
        order_key = f'{key[0]}/B{key[1]}/{key[2]}/rho{key[3]}'
        if order_key not in plan['arm_orders']:
            rotation = (i//2)%3
            order = list(audit.ARMS[rotation:]+audit.ARMS[:rotation])
            plan['arm_orders'][order_key] = [order,order[::-1]]
        order = plan['arm_orders'][order_key][key[4]]
        for arm in order:
            previous_arm = 'native' if arm=='native' else 'fast'
            row = copy.deepcopy(old_index[(*key,previous_arm)])
            row['arm'] = arm
            row['arm_order'] = order[:]
            row.pop('arrival_fix_source_sha256')
            row['arrival_combo_source_sha256'] = 'synthetic-combo-only'
            row['arrival_combo_schema_version'] = 1
            if arm=='combined':
                service = round((row['coordinator_finish_ns'][0]-row['coordinator_start_ns'][0])*.9)
                row.update(row_clocks(row['arrival_schedule'],service))
                row['assignment_sha256'] = 'different-online-assignments'
            row.update(dispatch_mode='online' if arm=='combined' else 'native',
                noop_enabled=arm!='native',assignment_selection_cost_included=True,
                online_residency_metadata_verified=arm=='combined',
                mapper_calls_measured=256 if arm=='combined' else 0,
                rank_gpu_uuids=plan['hardware']['gpu_uuids'][:])
            row['queue_metrics'] = audit.clocks.timing_metrics(row)
            row['observed_window_spans'] = audit.clocks.window_spans(row)
            rows.append(row)
    plan['freeze'] = {'sha256':audit.digest({k:v for k,v in plan.items() if k!='freeze'})}
    for row in rows:row['config_hash'] = plan['freeze']['sha256']
    return plan,rows


class IndependentComboInferenceTests(unittest.TestCase):
    def test_incremental_gate_cannot_be_rescued_by_large_native_gain(self):
        result,exact = audit.inference(triplets_fixture(lambda *_:1.,baseline=.5),strata_fixture())
        self.assertEqual(result['incremental_acceptance'],'not_established')
        self.assertEqual(result['acceptance'],'not_established')
        secondary = result['secondary_baseline']
        self.assertAlmostEqual(secondary['ratio'],.5)
        self.assertEqual(secondary['status'],'not_tested_gate_closed')
        for key in ('passes','ci95','upper_confidence_bound'):self.assertIsNone(secondary[key])
        self.assertNotIn('high_load_mean_response_baseline',exact)

    def test_small_batch_guard_prevents_large_batch_from_hiding_regression(self):
        result,_ = audit.inference(triplets_fixture(lambda block,batch,*_:1.03 if batch==128 else .80),strata_fixture())
        self.assertTrue(result['primary']['passes'])
        self.assertEqual(result['guards'][2]['name'],'b128_high_load_mean_response')
        self.assertFalse(result['guards'][2]['passes'])
        self.assertFalse(result['secondary_baseline']['gate_passed'])
        self.assertEqual(result['incremental_acceptance'],'not_established')

    def test_three_guard_bonferroni_quantiles_and_exact_count_distribution(self):
        offsets = [0,.002,.009,.03,.018,.051,.072,.12]
        result,exact = audit.inference(triplets_fixture(lambda block,*_:math.exp(-.2+offsets[int(block[-1])])),strata_fixture())
        names,groups,draws = audit.bootstrap_draws(strata_fixture())
        for guard in result['guards']:
            self.assertEqual(guard['guard_family_size'],3)
            self.assertEqual(guard['one_sided_confidence_level'],1-.05/3)
            values = np.asarray([guard['block_log_ratios'][name] for name in names])
            expected = np.quantile(np.exp(values[draws].mean(axis=1)),1-.05/3,method='linear')
            self.assertAlmostEqual(guard['upper_confidence_bound'],expected,places=14)
            support = exact[guard['name']]
            self.assertEqual(support['count_combinations'],81)
            self.assertEqual(sum(x['multiplicity'] for x in support['support']),256)
            self.assertEqual(sum(x['monte_carlo_draws'] for x in support['support']),100000)
            expanded = sorted(x['ratio'] for x in support['support'] for _ in range(x['multiplicity']))
            self.assertAlmostEqual(support['exact_requested_upper_quantile'],expanded[math.ceil((1-.05/3)*256)-1])

    def test_gate_open_still_requires_secondary_practical_success(self):
        result,_ = audit.inference(triplets_fixture(baseline=.97),strata_fixture())
        self.assertEqual(result['incremental_acceptance'],'pass')
        self.assertTrue(result['secondary_baseline']['confirmatory'])
        self.assertFalse(result['secondary_baseline']['passes'])
        self.assertEqual(result['acceptance'],'not_established')

    def test_repeat_log_average_not_arithmetic_ratio_and_no_pseudoreplication(self):
        rows = triplets_fixture(lambda block,batch,pattern,rho,rep,metric:(.64,1.)[rep])
        result,_ = audit.inference(rows,strata_fixture())
        self.assertAlmostEqual(result['primary']['ratio'],.8,places=14)
        self.assertNotAlmostEqual(result['primary']['ratio'],.82,places=10)
        for altered in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
            with self.assertRaises(AssertionError):audit.inference(altered,strata_fixture())


class IndependentComboRawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan,cls.rows = full_fixture()
        cls.summary,_ = audit.recompute(cls.plan,cls.rows)

    def test_384_records_128_triplets_384_contrasts_and_distinct_online_routes(self):
        result = audit.verify(self.plan,self.rows,self.summary)
        self.assertTrue(result['valid'])
        self.assertEqual(result['matched_triplets'],128)
        self.assertEqual(result['pairs_per_contrast'],{k:128 for k in audit.CONTRASTS})
        self.assertAlmostEqual(result['confirmatory']['primary']['ratio'],.9,places=14)
        self.assertEqual(sum(result['arm_position_counts']['combined']),128)

    def test_wrong_statistics_or_family_count_or_changed_order_rejected(self):
        for mutation in ('ratio','guard_quantile','guard_family','draw_sha','paired_ratio','position_count'):
            summary = copy.deepcopy(self.summary)
            if mutation=='ratio':summary['confirmatory']['primary']['ratio'] += .01
            elif mutation=='guard_quantile':summary['confirmatory']['guards'][0]['one_sided_confidence_level'] = .975
            elif mutation=='guard_family':summary['confirmatory']['guards'][1]['guard_family_size'] = 2
            elif mutation=='draw_sha':summary['confirmatory']['bootstrap']['draw_indices_sha256'] = 'wrong'
            elif mutation=='paired_ratio':summary['paired_cases'][0]['log_ratios']['response_mean_ns'] += .1
            else:summary['arm_position_counts']['combined'][1] += 1
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):
                audit.verify(self.plan,self.rows,summary)

    def test_wrong_dispatch_omitted_job_or_altered_frozen_order_rejected(self):
        for mutation in ('calls','assignment','logical','job','order','rank_gpu'):
            rows = self.rows[:]
            target = next(i for i,row in enumerate(rows) if row['arm']=='shortcut')
            rows[target] = copy.deepcopy(rows[target])
            if mutation=='calls':rows[target]['mapper_calls_measured'] = 256
            elif mutation=='assignment':rows[target]['assignment_sha256'] = 'changed'
            elif mutation=='logical':rows[target]['decision_trajectory_sha256'] = 'changed'
            elif mutation=='job':rows[target]['coordinator_finish_ns'].pop()
            elif mutation=='rank_gpu':rows[target]['rank_gpu_uuids'][0] = 'wrong'
            else:rows[target]['arm_order'] = rows[target]['arm_order'][::-1]
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):
                audit.verify(self.plan,rows,self.summary)

    def test_missing_or_duplicate_arm_trial_rejected(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[0]],self.rows[:-1]+[self.rows[0]]):
            with self.assertRaises(AssertionError):audit.verify(self.plan,rows,self.summary)

    def test_gate_closed_report_cannot_publish_secondary_interval_or_extra_test(self):
        rows = self.rows[:]
        shortcut = {tuple(r[k] for k in audit.UNIT):r for r in rows if r['arm']=='shortcut'}
        for i,row in enumerate(rows):
            if row['arm']!='combined':continue
            control = shortcut[tuple(row[k] for k in audit.UNIT)]
            service = control['coordinator_finish_ns'][0]-control['coordinator_start_ns'][0]
            rows[i] = copy.deepcopy(row)
            rows[i].update(row_clocks(row['arrival_schedule'],service))
            rows[i]['queue_metrics'] = audit.clocks.timing_metrics(rows[i])
            rows[i]['observed_window_spans'] = audit.clocks.window_spans(rows[i])
        summary,_ = audit.recompute(self.plan,rows)
        self.assertEqual(summary['confirmatory']['secondary_baseline']['status'],'not_tested_gate_closed')
        self.assertTrue(audit.verify(self.plan,rows,summary)['valid'])
        for mutation in ('interval','extra_test'):
            altered = copy.deepcopy(summary)
            secondary = altered['confirmatory']['secondary_baseline']
            if mutation=='interval':secondary.update(ci95=[.5,.6],upper_confidence_bound=.6,passes=True,confirmatory=True)
            else:secondary['unplanned_ci99'] = [.5,.7]
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):
                audit.verify(self.plan,rows,altered)


if __name__=='__main__':unittest.main()
