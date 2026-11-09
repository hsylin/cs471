#!/usr/bin/env python3
"""Resolve a payback template from immutable capture/composition inventories.

This consumes provenance only, never timing observations. The separate freeze
command adds the manifest hash after the resolved configuration is reviewable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from followup.payback import followup_fingerprint
from followup.payback_analysis import validate_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--capture-index', type=Path, required=True)
    p.add_argument('--replay-index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    manifest = json.loads(args.template.read_text())
    capture = json.loads(args.capture_index.read_text())
    replay = json.loads(args.replay_index.read_text())
    if replay['schema'] != 'qwen_router_replay_index_v2':
        raise ValueError('Require explicit v2 one-transient composition index')
    banks = {b['block_id']: b for b in capture}
    if len(banks) != len(capture) or not all(b['correctness_pass'] for b in capture):
        raise ValueError('Duplicate or unverified capture')
    digest, files = followup_fingerprint()
    manifest['environment']['followup_sha256'] = digest
    manifest['required_record_metadata']['followup_sha256'] = digest
    manifest['followup_source_files'] = files
    manifest['composition_provenance'] = {k:v for k,v in replay.items() if k != 'rows'}
    manifest['tapes'] = []
    for row in replay['rows']:
        if row['block_id'] not in manifest['matrix']['block_ids']:
            continue
        bank = banks[row['block_id']]
        if row['split'] != manifest['split'] or bank['split'] != manifest['split']:
            raise ValueError('Capture/replay split mismatch')
        if row['capture_sha256'] != bank['capture_sha256']:
            raise ValueError('Replay refers to a different capture')
        manifest['tapes'].append({
            'block_id':row['block_id'], 'regime':row['regime'],
            'batch_tokens':row['batch_tokens'],
            'capture_path':'banks/'+Path(bank['bank_file']).name,
            'replay_path':'tapes-v2/'+Path(row['path']).name,
            'capture_sha256':bank['capture_sha256'], 'replay_sha256':row['sha256']})
    if args.smoke:
        if manifest['split'] != 'pilot':
            raise ValueError('Smoke must use pilot sources')
        manifest['name'] = 'payback-correctness-smoke'
        manifest['study_role'] = 'smoke'
        manifest['matrix']['candidate_ids'] = ['native', 'one_copy']
    validate_manifest(manifest, frozen=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    print(json.dumps({'resolved':str(args.output), 'followup_sha256':digest,
                      'tapes':len(manifest['tapes'])}))


if __name__ == '__main__':
    main()
