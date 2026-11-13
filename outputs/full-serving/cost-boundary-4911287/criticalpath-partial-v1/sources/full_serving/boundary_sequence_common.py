"""CPU-only shared reconstruction helpers copied from admitted cost orchestrator285f32e."""
import hashlib,json,math
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def write_new(path, value):
    with Path(path).open("x") as f: f.write(json.dumps(value, indent=2)+"\n")

def arithmetic(records):
    assert records and all(r["success"] for r in records)
    def dist(values):
        assert all(math.isfinite(x) and x >= 0 for x in values)
        values = sorted(values); position = (len(values)-1)*.95
        lower = int(position); upper = min(lower+1,len(values)-1)
        return {"mean":math.fsum(values)/len(values),
                "p95":values[lower]+(position-lower)*(values[upper]-values[lower])}
    assert all(r["first_content_ns"] is not None for r in records)
    return {"requests":len(records), "failed_requests":0,
            "response_ms":dist([(r["end_ns"]-r["scheduled_ns"])/1e6 for r in records]),
            "ttft_ms":dist([(r["first_content_ns"]-r["scheduled_ns"])/1e6 for r in records]),
            "output_tokens":sum(r["usage"]["completion_tokens"] for r in records),
            "ceiling_hits":sum(r["usage"]["completion_tokens"]==8192 for r in records)}

def verify_arithmetic(value, summary):
    for local, remote in (("response_ms","response_from_schedule_ms"),("ttft_ms","ttft_from_schedule_ms")):
        for metric in ("mean","p95"):
            assert math.isclose(value[local][metric], summary[remote][metric], rel_tol=1e-12, abs_tol=1e-9)

def runtime_signature(identity, checkpoint, checkpoint_sha):
    assert checkpoint["complete"] is True and checkpoint["kind"]=="full_checkpoint_sha256"
    assert len(identity)==2
    assert all(r["checkpoint_manifest_sha256"]==checkpoint_sha for r in identity)
    # The manifest includes a per-check timestamp. Verify each exact digest first,
    # then compare every content field except that timestamp across sessions.
    return json.loads(json.dumps({"runtime":[{k:v for k,v in role.items() if k!="checkpoint_manifest_sha256"} for role in identity],
            "checkpoint":{k:v for k,v in checkpoint.items() if k!="checked_at"}}))

def ratio(left,right):
    assert left>=0 and right>0
    return {"ratio":left/right,"change_percent":100*(left/right-1)}

def raw_hashes(run):
    # PRIVATE receipt only. Logs and raw records must never be published.
    return {str(p.relative_to(run)):sha(p) for p in sorted(Path(run).glob("*/results/**/*")) if p.is_file()}
