"""Verify public aggregate arithmetic and source bytes, without private records."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    v = read('final-summary.json')
    assert len(v['rows']) == 6
    assert sum(r['requests'] for r in v['rows']) == 3072
    for ds, comparisons in v['comparisons'].items():
        cells = {r['treatment']: r for r in v['rows'] if r['dataset'] == ds}
        assert set(cells) == {'off', 'b4', 'b4_noop'}
        for r in cells.values():
            assert r['requests'] == 512 and r['failures'] == 0
            assert r['output_tokens'] == {'longbench':75246, 'numinamath':741753}[ds]
            assert math.isclose(r['elapsed_s']-r['offered_horizon_s'], r['drain_s'], abs_tol=1e-9)
        for pair, c in comparisons.items():
            a,b = (cells[t] for t in pair.split('/'))
            gain = 100*(1-a['mean_response_ms']/b['mean_response_ms'])
            assert math.isclose(gain,c['mean_response_gain_percent'],abs_tol=1e-10)
            assert c['useful_mean_gain_at_least5percent'] == (gain >= 5)
            assert c['mean_response_regression_exceeds2percent'] == (gain < -2)
            for metric in ['response_ms','ttft_ms']:
                for stat in ['mean','p95']:
                    ratio = a[stat+'_'+metric]/b[stat+'_'+metric]
                    assert math.isclose(ratio,c['metrics'][metric][stat]['ratio'],abs_tol=1e-12)
    manifest = read('source-manifest.json')
    base = manifest['snapshots']['final']['files']
    for label, snap in manifest['snapshots'].items():
        for name, digest in snap['files'].items():
            p = ROOT/'sources'/(label if name in snap['stored_files'] else 'final')/name
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert sha(p) == digest, (label,name)
        reconstructed = {**{n:h for n,h in base.items() if n not in snap['removed_from_base']},**snap['stored_files']}
        assert reconstructed == snap['files'], label
    assert sha(ROOT/'final-protocol.json') == v['protocol_sha256']
    if (ROOT/'publication-manifest.json').exists():
        for name,digest in read('publication-manifest.json')['files'].items():
            assert sha(ROOT/name) == digest, name
    print('PASS: 6 aggregate rows, ratios, frozen protocol and 9 source snapshots; private raw data not re-audited by this script.')

def materialize(label, destination):
    snap = read('source-manifest.json')['snapshots'][label]
    assert not destination.exists(), 'Use a new directory.'
    destination.mkdir(parents=True)
    for name,digest in snap['files'].items():
        src = ROOT/'sources'/(label if name in snap['stored_files'] else 'final')/name
        assert sha(src) == digest
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target)
    print('Materialized source subset:',destination)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--materialize',choices=list(read('source-manifest.json')['snapshots']))
    p.add_argument('--destination',type=Path)
    a=p.parse_args();verify()
    if a.materialize:
        assert a.destination is not None
        materialize(a.materialize,a.destination)
