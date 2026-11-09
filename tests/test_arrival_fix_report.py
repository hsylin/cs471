"""Synthetic CPU reporting checks; fixtures are not experimental observations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from arrival_fix.analysis import analyze
from tests.test_arrival_fix_analysis import full_fixture

SPEC = importlib.util.spec_from_file_location('arrival_fix_report', Path(__file__).resolve().parents[1] / 'scripts/report_arrival_fix.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class ArrivalFixReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan, cls.rows = full_fixture()
        cls.summary = analyze(plan, cls.rows)

    def test_all_cells_seed_ratios_confirmation_scope_and_raw_units(self):
        cells, seeds = REPORT.prepare_summary(self.summary)
        self.assertEqual((len(cells), len(seeds)), (8, 64))
        self.assertAlmostEqual(cells[0]['response_mean_ns_fast_over_native'], .9)
        self.assertAlmostEqual(cells[0]['native_service_mean_ms'], .001)
        confirmation = REPORT.confirmation_rows(self.summary)
        self.assertEqual(len(confirmation), 3)
        self.assertIsNotNone(confirmation[0]['lower_ci95'])
        self.assertIsNone(confirmation[1]['lower_ci95'])
        self.assertIsNone(confirmation[2]['lower_ci95'])
        steps = REPORT.raw_step_rows(self.summary, self.rows)
        self.assertEqual(len(steps), 512)
        self.assertEqual(steps[0]['trials_per_arm'], 64)
        self.assertAlmostEqual(steps[0]['native_mean_service_ms'], .001)
        self.assertAlmostEqual(steps[0]['fast_mean_service_ms'], .0009)

    def test_partial_changed_or_mixed_evidence_is_rejected_before_rendering(self):
        for mutation in ('invalid', 'missing_cell', 'duplicate_pair', 'wrong_cell_ratio'):
            summary = copy.deepcopy(self.summary)
            if mutation == 'invalid': summary['valid'] = False
            elif mutation == 'missing_cell': summary['descriptive_cells'].pop()
            elif mutation == 'duplicate_pair': summary['paired_cases'][1] = summary['paired_cases'][0]
            else: summary['descriptive_cells'][0]['ratios']['response_mean_ns'] = .123
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                REPORT.prepare_summary(summary)
        with self.assertRaises(ValueError):
            REPORT.raw_step_rows(self.summary, self.rows[:-1])
        with tempfile.TemporaryDirectory(prefix='arrival-report-fixture-') as folder:
            path = Path(folder)
            summary = copy.deepcopy(self.summary)
            summary['records_file_sha256'] = 'wrong-raw-source'
            (path/'summary.json').write_text(json.dumps(summary))
            (path/'records.jsonl').write_text('\n'.join(json.dumps(row) for row in self.rows)+'\n')
            with self.assertRaises(ValueError):
                REPORT.report(path/'summary.json', path/'output', path/'records.jsonl')
            self.assertFalse((path/'output').exists())


if __name__ == '__main__':
    unittest.main()
