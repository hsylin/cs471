"""Synthetic report fixtures test completeness and units, not GPU performance."""
import copy
import importlib.util
import itertools
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('arrival_report', Path(__file__).resolve().parents[1] / 'scripts/report_instructor_arrivals.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def summary_fixture():
    cells = []
    for block, domain, batch, pattern, rho in itertools.product(('fixture-a', 'fixture-b'), ('domain0', 'domain1'),
            (128, 1024), ('constant', 'burst8'), (.5, .85)):
        contrasts = {source: {'native': 2_000_000, 'online': 1_000_000} for source, _, _, _ in REPORT.METRICS}
        contrasts['backlog_at_horizon'] = {'native': 2., 'online': 1.}
        contrasts['coordinator_server_busy_fraction'] = {'native': .5, 'online': .4}
        cells.append(dict(block_id=block, regime=domain, batch_tokens=batch, pattern=pattern,
                          rho=rho, technical_repetitions=2, contrasts=contrasts))
    return dict(valid=True, phase='evaluation', expected_records=128, observed_records=128, valid_records=128,
                allocation_ids=['synthetic-fixture'], repeat_averaged_cases=cells)


class ArrivalReportTests(unittest.TestCase):
    def test_exact_cells_and_unit_conversion_preserve_supplied_arm_values(self):
        rows = REPORT.table_rows(summary_fixture())
        self.assertEqual(len(rows), 32)
        self.assertEqual(rows[0]['native_response_mean_ms'], 2.)
        self.assertEqual(rows[0]['online_response_p95_ms'], 1.)
        self.assertEqual(rows[0]['native_backlog_at_horizon_jobs'], 2.)
        self.assertEqual(rows[0]['online_coordinator_busy_fraction'], .4)
        table = REPORT.markdown_table(rows)
        self.assertEqual(len(table.splitlines()), 34)

    def test_invalid_partial_duplicate_or_nonfinite_summary_is_rejected(self):
        for mutation in ('invalid', 'missing', 'duplicate', 'missing_trial', 'nonfinite', 'busy_out_of_range'):
            summary = summary_fixture()
            if mutation == 'invalid':
                summary['valid'] = False
            elif mutation == 'missing':
                summary['repeat_averaged_cases'].pop()
            elif mutation == 'duplicate':
                summary['repeat_averaged_cases'][1] = copy.deepcopy(summary['repeat_averaged_cases'][0])
            elif mutation == 'missing_trial':
                summary['observed_records'] = 127
            elif mutation == 'nonfinite':
                summary['repeat_averaged_cases'][0]['contrasts']['response_mean_ns']['online'] = float('nan')
            else:
                summary['repeat_averaged_cases'][0]['contrasts']['coordinator_server_busy_fraction']['online'] = 1.1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                REPORT.table_rows(summary)


if __name__ == '__main__':
    unittest.main()
