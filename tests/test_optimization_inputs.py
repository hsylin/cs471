"""Fresh corpus isolation and exact reuse of the authenticated capture computation."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from optimization.input_sources import ROOT, source_text, validate_source_manifest


def manifest():
    return json.loads((ROOT/'optimization/corpus/sources.json').read_text())


def test_fresh_documents_exclude_every_prior_source():
    result = validate_source_manifest(manifest())
    assert result['source_document_counts'] == {'pilot': 2, 'final': 8}
    assert result['source_documents'] == result['prior_documents_excluded'] == 20
    assert result['overlap_by_id_url_raw_or_cleaned_content'] is False


@pytest.mark.parametrize('field', ['source_id', 'url', 'sha256', 'cleaned_text_sha256'])
def test_renaming_cannot_hide_prior_document_reuse(field):
    data = manifest()
    old = json.loads((ROOT/'followup/corpus/sources.json').read_text())['blocks'][0]['sources'][0]
    if field == 'cleaned_text_sha256':
        import hashlib
        value = hashlib.sha256(source_text(old, ROOT/'followup/corpus').encode()).hexdigest()
    else:
        value = old[field]
    data['blocks'][0]['sources'][0][field] = value
    with pytest.raises(ValueError, match='overlap'):
        validate_source_manifest(data)


def test_mislabeled_split_is_rejected():
    data = manifest()
    data['blocks'][0]['split'] = 'final'
    with pytest.raises(ValueError, match='split/block'):
        validate_source_manifest(data)


def test_capture_model_and_router_computation_is_exact_predecessor():
    old = (ROOT/'scripts/capture_router_tapes.py').read_text()
    new = (ROOT/'scripts/capture_optimization_tapes.py').read_text()
    # Checkpoint loading and native model construction remain byte-for-byte equal.
    for start, end in [('    config_path, index_path =', '    index_file ='),
                       ('        for domain, source in enumerate', '        arrays =')]:
        assert old[old.index(start):old.index(end)] == new[new.index(start):new.index(end)]


def test_composer_rejects_nonempty_output_before_mutation(tmp_path):
    path = ROOT/'scripts/compose_optimization_tapes.py'
    spec = importlib.util.spec_from_file_location('compose_optimization_tapes', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path/'existing';out.mkdir();(out/'marker').write_text('unchanged')
    with pytest.raises(FileExistsError):
        module.compose_directory(tmp_path/'banks', out)
    assert (out/'marker').read_text() == 'unchanged'
