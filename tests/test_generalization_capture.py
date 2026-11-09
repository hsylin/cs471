import json
import pytest
from generalization.data import digest,deduplicate
from followup.router_tapes import MODEL,REVISION
from scripts.capture_generalization_inputs import validate_sources


def test_capture_validates_manifest_before_importing_gpu_runtime(tmp_path):
    rows,_=deduplicate([dict(dataset='mbpp',source_id='one',text='source one')])
    m=dict(schema='generalization_sources_v1',source_model=MODEL,source_revision=REVISION,blocks=[dict(sources=rows)])
    m['content_sha256']=digest(m)
    p=tmp_path/'sources.json';p.write_text(json.dumps(m))
    assert validate_sources(p)==m
    m['blocks'][0]['sources'][0]['text']='changed';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Changed/unfrozen'):validate_sources(p)
    m['content_sha256']=digest({k:v for k,v in m.items() if k!='content_sha256'});p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Text changed'):validate_sources(p)
