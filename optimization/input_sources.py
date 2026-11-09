"""CPU provenance checks for the new, document-disjoint optimization corpus."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PRIOR_SOURCES = ROOT / 'followup/corpus/sources.json'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_text(source: dict, directory: Path) -> str:
    path = directory / source['file']
    if sha256(path) != source['sha256']:
        raise ValueError(f'Source checksum mismatch: {path}')
    text = path.read_text(encoding='utf-8-sig')
    if source['cleaning'] == 'content_between_gutenberg_start_end_markers':
        start = re.search(r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*', text)
        end = re.search(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*', text)
        if start is None or end is None or end.start() <= start.end():
            raise ValueError('Unrecognized Gutenberg boundaries')
        text = text[start.end():end.start()].strip()
    elif source['cleaning'] != 'none':
        raise ValueError('Unknown source cleaning rule')
    expected = source.get('cleaned_text_sha256')
    if expected and hashlib.sha256(text.encode()).hexdigest() != expected:
        raise ValueError('Cleaned source checksum mismatch')
    return text


def validate_source_manifest(manifest: dict, directory: Path | None = None) -> dict:
    directory = directory or ROOT / 'optimization/corpus'
    if manifest.get('study_id') != 'kernel_optimization_v1':
        raise ValueError('Expected distinct optimization study identity')
    if manifest.get('excluded_source_manifest_sha256') != sha256(PRIOR_SOURCES):
        raise ValueError('Prior source-manifest pin changed')
    prior = json.loads(PRIOR_SOURCES.read_text())
    old = [s for b in prior['blocks'] for s in b['sources']]
    previous = {
        'source_id': {s['source_id'] for s in old},
        'url': {s['url'] for s in old},
        'sha256': {s['sha256'] for s in old},
        'cleaned_text_sha256': {
            hashlib.sha256(source_text(s, PRIOR_SOURCES.parent).encode()).hexdigest()
            for s in old},
    }
    seen = {key: set() for key in previous}
    counts = {'pilot': 0, 'final': 0}
    blocks = set()
    for block in manifest['blocks']:
        if block['block_id'] in blocks:
            raise ValueError('Repeated source block')
        blocks.add(block['block_id'])
        if block['split'] not in counts or not block['block_id'].startswith('opt-'+block['split']+'-'):
            raise ValueError('Expected explicit optimization split/block identity')
        counts[block['split']] += 1
        if [s['domain'] for s in block['sources']] != ['prose', 'code']:
            raise ValueError('Each block must contain prose then code')
        for source in block['sources']:
            for field in seen:
                value = source[field]
                if value in seen[field] or value in previous[field]:
                    raise ValueError(f'Source document overlap detected by {field}')
                seen[field].add(value)
            text = source_text(source, directory)
            if len(text) < 5000:
                raise ValueError('Source too short; require genuine full document')
            if source['domain'] == 'prose':
                metadata_path = directory / source['metadata_file']
                if sha256(metadata_path) != source['metadata_sha256']:
                    raise ValueError('Gutenberg metadata checksum mismatch')
                if 'Public domain in the USA' not in metadata_path.read_text():
                    raise ValueError('Archived catalog lacks declared copyright status')
            elif source.get('source_revision') != manifest['cpython_revision']:
                raise ValueError('CPython revision differs from manifest')
    if counts != {'pilot': 2, 'final': 8}:
        raise ValueError('Expected two pilot/eight final source pairs')
    if sha256(directory/'raw/cpython_LICENSE') != manifest['cpython_license_sha256']:
        raise ValueError('CPython license checksum mismatch')
    return {'study_id': manifest['study_id'], 'source_blocks': len(blocks),
            'source_documents': len(seen['source_id']), 'prior_documents_excluded': len(old),
            'overlap_by_id_url_raw_or_cleaned_content': False,
            'source_document_counts': counts, 'all_source_hashes_verified': True}
