"""Independent synthetic CPU fixtures; no production analysis imports/evidence."""
import copy
import gzip
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import unittest
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('independent_arrival_statistics',
    ROOT/'scripts/verify_arrival_fix_statistics.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def strata_fixture():
    # Deliberately interleaved names: grouping must not assume adjacent IDs.
    return {f'calendar-{i}':(f'source-{i%2}', f'domain{(i//2)%2}') for i in range(8)}


def pairs_fixture(log_value=lambda block, batch, pattern, rho, rep: math.log(.9)):
    return [dict(arrival_block_id=block, batch_tokens=batch, pattern=pattern, rho=rho,
        repetition=rep, log_ratios={metric:log_value(block,batch,pattern,rho,rep)
            for metric in audit.METRICS})
        for block,batch,pattern,rho,rep in itertools.product(strata_fixture(),
            (128,1024), ('spread','cluster'), (.5,.85), (0,1))]


def row_clocks(schedule, service):
    epoch = 10**18  # Rounding absolute times to float would lose nanoseconds.
    starts = [epoch+t for t in schedule['arrival_offsets_ns']]
    finishes = [t+service for t in starts]
    return dict(n_jobs=256, epoch_monotonic_ns=epoch, horizon_ns=schedule['horizon_ns'],
        arrival_offsets_ns=schedule['arrival_offsets_ns'][:],
        coordinator_admit_ns=starts[:], coordinator_start_ns=starts[:],
        coordinator_finish_ns=finishes, per_rank_start_ns=[starts[:],starts[:]],
        per_rank_gpu_complete_ns=[finishes[:],finishes[:]],
        observation_end_ns=epoch+schedule['horizon_ns'])


def full_fixture():
    blocks = [dict(id=name, seed=100+i, block_id=s[0], regime=s[1])
        for i,(name,s) in enumerate(strata_fixture().items())]
    plan = dict(phase='final', arrival_fix_source_sha256='synthetic-only',
        analysis=audit.SETTINGS.copy(), hardware=dict(hostname='fixture-host',
            gpu_uuids=['fixture-gpu1','fixture-gpu0'], world_size=2),
        matrix=dict(arrival_blocks=blocks, batches=[128,1024], patterns=['spread','cluster'],
            rhos=[.5,.85], reps=2, arms=['native','fast']), calibration={}, schedules={})
    for block,batch in itertools.product(blocks,(128,1024)):
        plan['calibration'][f"{block['block_id']}/{block['regime']}/B{batch}"] = {'allocation_id':'fixture-allocation'}
        entries = []
        for pattern,rho in itertools.product(('spread','cluster'),(.5,.85)):
            entry = dict(pattern=pattern,rho=rho,horizon_ns=256*10000,
                arrival_offsets_ns=[10000*i for i in range(256)])
            entry['schedule_sha256'] = audit.digest(entry)
            entries.append(entry)
        plan['schedules'][f"{block['id']}/B{batch}"] = entries
    plan['freeze'] = {'sha256':audit.digest(plan)}
    rows = []
    for block,batch,pattern,rho,rep in itertools.product(blocks,(128,1024),('spread','cluster'),(.5,.85),(0,1)):
        schedule = next(x for x in plan['schedules'][f"{block['id']}/B{batch}"]
                        if (x['pattern'],x['rho']) == (pattern,rho))
        order = ['native','fast'] if rep == 0 else ['fast','native']
        for arm in order:
            # Heterogeneous block ratios produce nondegenerate block uncertainty.
            service = 1000 if arm == 'native' else 850+20*(block['seed']-100)
            row = row_clocks(schedule,service)
            row.update(arrival_block_id=block['id'], arrival_seed=block['seed'],
                block_id=block['block_id'], regime=block['regime'], batch_tokens=batch,
                pattern=pattern,rho=rho,repetition=rep,arm=arm,arm_order=order[:],
                status='complete',phase='final',config_hash=plan['freeze']['sha256'],
                arrival_fix_source_sha256='synthetic-only',correctness_pass=True,evidence_eligible=True,
                diagnostic_only=False,guard_symbols_restored=True,normal_completion_verified=True,
                resident_state_verified=True,rank_assignment_agreement=True,migration_bytes=0,
                migration_sent_messages=0,terminal_proposals_suppressed=1,policy='SKIP',
                preflight_correctness={'correctness_pass':True},source_map=[0,1],final_map=[0,1],
                physical_gpu_uuids=['fixture-gpu0','fixture-gpu1'],hostname='fixture-host.example',
                allocation_id='fixture-allocation',arrival_schedule=copy.deepcopy(schedule),
                arrival_clock_contract=dict(monotonic=True,adjustable=False,implementation='CLOCK_MONOTONIC'),
                capture_sha256='capture',replay_sha256='replay',source_prompt_ids=['doc1','doc2'],
                source_weights_sha256='weights',source_weight_rank_sha256=['r0','r1'],resolved_kernel_config={},
                logical_routing_sha256='routes',decision_trajectory_sha256='trajectory',
                warmup_raw_indices=list(range(16)),conditioning_raw_indices=list(range(16,48))*2,
                measured_raw_indices=list(range(16,48))*8,assignment_sha256='assignment')
            guard = dict(mode='fast',diagnostic_timing=False,pending_pair=False,movement_calls=4,
                commit_calls=4,agreement_calls=4,local_unchanged_calls=4,unanimously_unchanged_calls=4,
                movement_skips=4,commit_skips=4,original_movement_calls=0,original_commit_calls=0,
                fallback_calls=0,diagnostics=[])
            row['noop_guard_ranks'] = [guard.copy(),guard.copy()] if arm == 'fast' else []
            row['queue_metrics'] = audit.clocks.timing_metrics(row)
            row['observed_window_spans'] = audit.clocks.window_spans(row)
            rows.append(row)
    return plan,rows


class IndependentInferenceTests(unittest.TestCase):
    def test_paired_logs_are_averaged_not_arithmetic_ratios(self):
        pairs = pairs_fixture(lambda block,batch,pattern,rho,rep:math.log((.64,1.)[rep]))
        result,_ = audit.inference(pairs,strata_fixture())
        self.assertAlmostEqual(result['primary']['ratio'],.8, places=14)
        self.assertNotAlmostEqual(result['primary']['ratio'],.82, places=8)
        self.assertEqual(result['primary']['arrival_blocks'],8)

    def test_stratum_fixed_effects_cannot_create_composition_uncertainty(self):
        strata = strata_fixture()
        values = dict(zip(sorted(set(strata.values())),(-.4,-.2,.1,.5)))
        result,exact = audit.inference(pairs_fixture(lambda block,*_:values[strata[block]]),strata)
        np.testing.assert_allclose(result['primary']['ci95'],[1.,1.],atol=1e-14,rtol=0)
        self.assertEqual(exact['high_load_mean_response']['count_combinations'],81)

    def test_exact_81_distribution_matches_256_independent_ordered_choices(self):
        strata = strata_fixture()
        pairs = pairs_fixture(lambda block,*_:int(block[-1])*.07-.3)
        result,exact = audit.inference(pairs,strata)
        names,groups,draws = audit.bootstrap_draws(strata)
        logs = result['primary']['block_log_ratios']
        ordered = []
        for choices in itertools.product(((0,0),(0,1),(1,0),(1,1)),repeat=4):
            selected = [names[members[c]] for members,pair in zip(groups,choices) for c in pair]
            ordered.append(math.exp(math.fsum(logs[b] for b in selected)/8))
        ordered.sort()
        support = exact['high_load_mean_response']
        self.assertEqual(sum(x['multiplicity'] for x in support['support']),256)
        self.assertEqual(sum(x['monte_carlo_draws'] for x in support['support']),100000)
        self.assertAlmostEqual(support['exact_percentile_025'],ordered[math.ceil(.025*256)-1])
        self.assertAlmostEqual(support['exact_percentile_975'],ordered[math.ceil(.975*256)-1])
        for j,members in enumerate(groups):
            self.assertTrue(np.isin(draws[:,j*2:j*2+2],members).all())
        self.assertEqual(draws.shape,(100000,8))

    def test_missing_repeat_and_extra_job_units_rejected(self):
        for alteration in ('missing','duplicate','job_units','unbalanced'):
            pairs, strata = pairs_fixture(),strata_fixture()
            if alteration == 'missing':pairs.pop()
            elif alteration == 'duplicate':pairs.append(copy.deepcopy(pairs[0]))
            elif alteration == 'job_units':
                pairs[0]['repetition'] = 255
            else:strata['calendar-0'] = strata['calendar-1']
            with self.subTest(alteration=alteration),self.assertRaises(AssertionError):
                audit.inference(pairs,strata)

    def test_primary_and_guard_decisions_match_exact_declared_inequalities(self):
        for ratio in (.949,.95,1.02,1.021):
            result,_ = audit.inference(pairs_fixture(lambda *_:math.log(ratio)),strata_fixture())
            self.assertEqual(result['primary']['passes'],result['primary']['upper_confidence_bound'] < .95)
            for guard in result['guards']:
                self.assertEqual(guard['passes'],guard['upper_confidence_bound'] <= 1.02)
            self.assertEqual(result['acceptance']=='pass',all(x['passes']
                for x in [result['primary'],*result['guards']]))


class IndependentRawStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan,cls.rows = full_fixture()
        cls.summary,_ = audit.recompute(cls.plan,cls.rows)

    def test_raw_complete_roundtrip_with_large_integer_epoch(self):
        result = audit.verify(self.plan,self.rows,self.summary)
        self.assertTrue(result['valid'])
        self.assertEqual(result['matched_pairs'],128)
        self.assertAlmostEqual(result['confirmatory']['primary']['ratio'],
            math.prod((850+20*i)/1000 for i in range(8))**(1/8),places=14)
        self.assertEqual(result['hardware']['hostname'],'fixture-host')

    def test_mutated_report_cannot_substitute_pooled_or_job_bootstrap_result(self):
        for kind in ('mean_ratios','zero_width_job_ci','draw_hash','repeat_count','pair','descriptive'):
            summary = copy.deepcopy(self.summary)
            if kind == 'mean_ratios':summary['confirmatory']['primary']['ratio'] = .92
            elif kind == 'zero_width_job_ci':
                summary['confirmatory']['primary']['ci95'] = [summary['confirmatory']['primary']['ratio']]*2
            elif kind == 'draw_hash':summary['confirmatory']['bootstrap']['draw_indices_sha256'] = 'wrong'
            elif kind == 'repeat_count':summary['confirmatory']['primary']['arrival_blocks'] = 16
            elif kind == 'pair':summary['paired_cases'][0]['log_ratios']['response_mean_ns'] += .01
            else:summary['descriptive_cells'][0]['ratios']['response_p95_ns'] += .01
            with self.subTest(kind=kind),self.assertRaises(AssertionError):
                audit.verify(self.plan,self.rows,summary)

    def test_missing_job_or_rank_completion_or_changed_calendar_fails_closed(self):
        for kind in ('job','rank','calendar','order','allocation','hardware','stored_metric'):
            rows = copy.deepcopy(self.rows)
            if kind == 'job':rows[0]['coordinator_finish_ns'].pop()
            elif kind == 'rank':rows[0]['per_rank_gpu_complete_ns'][1][0] += 1
            elif kind == 'calendar':rows[0]['arrival_offsets_ns'][2] += 1
            elif kind == 'order':rows[0],rows[1] = rows[1],rows[0]
            elif kind == 'allocation':rows[0]['allocation_id'] = 'other'
            elif kind == 'hardware':rows[0]['physical_gpu_uuids'][0] = 'other'
            else:rows[0]['queue_metrics']['summary_ns']['response']['mean'] += 1
            with self.subTest(kind=kind),self.assertRaises(AssertionError):
                audit.verify(self.plan,rows,self.summary)

    def test_duplicate_or_missing_trial_cannot_change_effective_unit(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[0]],self.rows[:-1]+[self.rows[0]]):
            with self.assertRaises(AssertionError):
                audit.verify(self.plan,rows,self.summary)

    def test_gzip_preserves_raw_summary_identity_but_content_mutation_fails(self):
        original = b'{"fixture": 1}\n{"fixture": 2}\n'
        summary = {'records_file_sha256':hashlib.sha256(original).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'records.jsonl.gz'
            path.write_bytes(gzip.compress(original,mtime=0))
            result = audit.records_identity(path,summary)
            self.assertEqual(result['records_decompressed_sha256'],summary['records_file_sha256'])
            self.assertNotEqual(result['records_file_sha256'],summary['records_file_sha256'])
            # Whitespace-only changes still invalidate byte identity.
            path.write_bytes(gzip.compress(original.replace(b': ',b':'),mtime=0))
            with self.assertRaisesRegex(AssertionError,'Summary raw-file identity'):
                audit.records_identity(path,summary)


if __name__ == '__main__':
    unittest.main()
