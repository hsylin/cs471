"""Exercise the actual Bash wrapper using fake Slurm commands, without GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess


def test_subset_launcher_passes_matching_dataset_and_report_arguments(tmp_path):
    root = tmp_path / 'repo'
    (root / 'full_serving').mkdir(parents=True)
    (root / 'scripts').mkdir()
    shutil.copy2(Path(__file__).parents[1] / 'full_serving/run_suite_allocation.sh', root / 'full_serving')
    (root / 'scripts/cluster-env.sh').write_text('export SKB_ROOT="$LAUNCH_TEST_ROOT"\n')
    cluster = tmp_path / 'cluster'
    (cluster / 'full-serving').mkdir(parents=True)
    bins = tmp_path / 'bin'
    bins.mkdir()
    record = tmp_path / 'commands.jsonl'
    commands = {
        'date': "print('2026-11-10T17:00:00+01:00')",
        'squeue': "import sys; print('RUNNING' if '%T' in sys.argv else 'mock[1-2]')",
        'scontrol': "print('mock1\\nmock2')",
        'srun': "import os,sys,json; open(os.environ['LAUNCH_TEST_RECORD'],'a').write(json.dumps(sys.argv[1:])+'\\n')",
    }
    for name, body in commands.items():
        p = bins / name
        p.write_text('#!/usr/bin/env python3\n' + body + '\n')
        p.chmod(0o755)
    env = dict(PATH=str(bins) + os.pathsep + os.environ['PATH'],
               LAUNCH_TEST_ROOT=str(cluster), LAUNCH_TEST_RECORD=str(record), FULL_SERVING_SOURCE_REVISION='a' * 40)
    selected = ['sharegpt', 'instructcoder', 'numinamath', 'longbench']
    subprocess.run(['bash', str(root / 'full_serving/run_suite_allocation.sh'), 'retry-v2', *selected], env=env, check=True)
    calls = [json.loads(s) for s in record.read_text().split('\n') if s]
    assert len(calls) == 2
    controller, reporter = calls
    assert controller[controller.index('--datasets') + 1:] == selected
    paths = reporter[reporter.index('full_serving.report_pilots') + 1:reporter.index('--output')]
    assert paths == [str(cluster / 'full-serving/retry-v2/results' / d) for d in selected + ['sharegpt-burstgpt-diagnostic']]
    assert not any('/wildchat' in arg for call in calls for arg in call)
    assert (cluster / 'full-serving/retry-v2/completed.txt').read_text() == 'suite_exit=0 report_exit=0\n'
