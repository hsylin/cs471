"""Synthetic CPU reporting contracts; these fixtures are not GPU observations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from arrival_combo.analysis import analyze, infer_endpoints
from tests.test_arrival_combo_analysis import full_fixture, pair_fixture, blocks_fixture

SPEC = importlib.util.spec_from_file_location('arrival_combo_report', Path(__file__).resolve().parents[1]/'scripts/report_arrival_combo.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class ArrivalComboReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan, cls.rows = full_fixture()
        cls.summary = analyze(plan, cls.rows)

    def test_three_contrasts_units_raw_service_and_closed_gate(self):
        cells, seeds = REPORT.prepare_summary(self.summary)
        self.assertEqual((len(cells), len(seeds)), (8, 192))
        self.assertAlmostEqual(cells[0]['response_mean_ns_combined_over_shortcut'], .84/.9)
        self.assertAlmostEqual(cells[0]['response_mean_ns_combined_over_native'], .84)
        self.assertAlmostEqual(cells[0]['native_service_mean_ms'], .001)
        confirmation = REPORT.confirmation_rows(self.summary)
        self.assertEqual(len(confirmation), 5)
        self.assertTrue(all(row['confirmatory_tested'] for row in confirmation))
        self.assertEqual(sum(row['lower_ci95'] is not None for row in confirmation), 2)
        steps, boundaries, architecture = REPORT.raw_descriptives(self.summary, self.rows)
        self.assertEqual((len(steps), len(boundaries), len(architecture)), (512, 18, 0))
        self.assertEqual(steps[0]['trials_per_arm'], 64)
        self.assertAlmostEqual(steps[0]['combined_mean_service_ms'], .00084)
        ordinary = next(row for row in boundaries if row['arm']=='native' and row['group']=='ordinary')
        self.assertEqual(ordinary['observations'], 64*252)
        self.assertAlmostEqual(ordinary['mean_trial_group_total_ms'], .252)
        summary = copy.deepcopy(self.summary)
        summary['confirmatory'] = infer_endpoints(pair_fixture(increment=.99, baseline=.5), blocks_fixture())
        secondary = REPORT.confirmation_rows(summary)[-1]
        self.assertFalse(secondary['confirmatory_tested'])
        self.assertIsNone(secondary['lower_ci95'])
        self.assertIsNone(secondary['upper_bound'])
        self.assertEqual(secondary['status'], 'not_tested_gate_closed')
        self.assertAlmostEqual(secondary['ratio'], .5)

    def test_incomplete_changed_or_mixed_sources_rejected_before_rendering(self):
        for mutation in ('invalid', 'missing_cell', 'duplicate_pair', 'wrong_ratio', 'missing_trial'):
            summary = copy.deepcopy(self.summary)
            if mutation=='invalid': summary['valid']=False
            elif mutation=='missing_cell': summary['descriptive_cells'].pop()
            elif mutation=='duplicate_pair': summary['paired_cases'][1]=summary['paired_cases'][0]
            elif mutation=='missing_trial': summary['trial_summaries'].pop()
            else: summary['descriptive_cells'][0]['contrasts']['combined_over_native']['ratios']['response_mean_ns']=.123
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): REPORT.prepare_summary(summary)
        with self.assertRaises(ValueError): REPORT.raw_descriptives(self.summary, self.rows[:-1])
        rows = copy.deepcopy(self.rows)
        rows[0]['arrival_combo_source_sha256']='wrong-execution-source'
        with self.assertRaises(ValueError): REPORT.raw_descriptives(self.summary, rows)
        with tempfile.TemporaryDirectory(prefix='combo-report-fixture-') as folder:
            path=Path(folder); summary=copy.deepcopy(self.summary)
            summary['records_file_sha256']='wrong-raw-file'
            (path/'summary.json').write_text(json.dumps(summary))
            (path/'records.jsonl').write_text('\n'.join(json.dumps(row) for row in self.rows)+'\n')
            with self.assertRaises(ValueError): REPORT.report(path/'summary.json', path/'output', path/'records.jsonl')
            self.assertFalse((path/'output').exists())


if __name__=='__main__': unittest.main()
