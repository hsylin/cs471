"""Synthetic harness tests; no experiment records are loaded."""
import importlib.util,json
from pathlib import Path
import pytest
p=Path(__file__).with_name('reproduce_stock_clean_cpu.py');spec=importlib.util.spec_from_file_location('clean_repro',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def pair(tmp_path):
 a={'valid':True,'n':216,'elapsed_ns':10**12,'ratio':.91,'record_files':[{'path':'server/a.jsonl','stored_bytes_sha256':'a','uncompressed_bytes_sha256':'b'}]}
 b={**a,'record_files':[{'path':'copy/a.jsonl.gz','stored_bytes_sha256':'c','uncompressed_bytes_sha256':'b'}]}
 x=tmp_path/'a.json';y=tmp_path/'b.json';x.write_text(json.dumps(a));y.write_text(json.dumps(b));return x,y,b

def test_only_path_and_compression_metadata_normalized(tmp_path):
 a,b,_=pair(tmp_path);assert m.compare_analysis(a,b)['passed']

@pytest.mark.parametrize('field,value',[('elapsed_ns',10**12+1),('n',215),('ratio',.92)])
def test_numerical_or_identity_changes_rejected(tmp_path,field,value):
 a,b,data=pair(tmp_path);data[field]=value;b.write_text(json.dumps(data))
 with pytest.raises(ValueError):m.compare_analysis(a,b)

def test_changed_decompressed_source_bytes_rejected(tmp_path):
 a,b,data=pair(tmp_path);data['record_files'][0]['uncompressed_bytes_sha256']='wrong';b.write_text(json.dumps(data))
 with pytest.raises(ValueError):m.compare_analysis(a,b)
