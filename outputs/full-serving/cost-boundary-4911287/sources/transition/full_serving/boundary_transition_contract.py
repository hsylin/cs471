"""Pilot-frozen candidate intervention contracts, separate from final policies."""
import hashlib
import json
from pathlib import Path

MODES=("keep","preinstalled","charged")
def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def validate_map(mapping):
    assert len(mapping)==48
    for row in mapping:
        assert len(row)==160 and all(type(x) is int and 0<=x<128 for x in row)
        assert set(row)==set(range(128))

def freeze_candidate(pilot):
    pilot=Path(pilot)
    audit=json.loads((pilot/"audit.json").read_text());assert audit["status"]=="passed"
    source=json.loads((pilot/"results/controller-source.json").read_text())["declared_git_revision"]
    peers=[];files={}
    for rank in range(8):
        p=pilot/"results/pilot/b4"/f"gate-events-rank{rank}.jsonl"
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        selected=[r for r in rows if r["kind"]=="decision"]
        assert all(r["rank"]==rank and r["mode"]=="b4" and r["rank_agreement"] for r in selected)
        assert len({r["step"] for r in selected})==len(selected)
        peers.append({r["step"]:r for r in selected})
        files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(p.keys()==peers[0].keys() for p in peers)
    for step in sorted(peers[0]):
        first=peers[0][step]
        for peer in peers[1:]:
            other=peer[step]
            for field in ("digest","old_map","admitted_map","native_candidate","previous_counts","recent_counts"):
                assert first[field]==other[field],(step,field)
        validate_map(first["old_map"]);validate_map(first["admitted_map"])
        if first["old_map"]!=first["admitted_map"]:
            break
    else:raise ValueError("Pilot contains no changed admitted candidate; intervention unavailable")
    spec=dict(schema="cs471_transition_candidate_v1",status="frozen_from_pilot",
              selection="First B4 rank-agreed admitted placement change in this predeclared allocation pilot; no performance ranking.",
              source_revision=source,pilot_audit_sha256=hashlib.sha256((pilot/"audit.json").read_bytes()).hexdigest(),
              source_files=files,source_step=step,gate_digest=first["digest"],
              old_map=first["old_map"],candidate_map=first["admitted_map"],
              old_map_sha256=canonical(first["old_map"]),candidate_map_sha256=canonical(first["admitted_map"]),
              pilot_demand_sha256=canonical([first["previous_counts"],first["recent_counts"]]),
              changed_slots=sum(a!=b for old,new in zip(first["old_map"],first["admitted_map"]) for a,b in zip(old,new)),
              online_trigger="First native proposal snapshot after all ranks observe measured-phase marker; no forced engine step or lookahead.",
              limitations="Same fixed candidate used for both datasets; not claimed optimal for either. Separate replays do not clone active KV state, generated tokens or routes. PREINSTALLED excludes migration from measured phase and changes preceding warmup state.")
    spec["seal"]=canonical(spec)
    return spec

def validate_spec(spec):
    assert spec["schema"]=="cs471_transition_candidate_v1" and spec["status"]=="frozen_from_pilot"
    assert spec["seal"]==canonical({k:v for k,v in spec.items() if k!="seal"})
    validate_map(spec["old_map"]);validate_map(spec["candidate_map"])
    assert spec["old_map"]!=spec["candidate_map"]
    assert canonical(spec["old_map"])==spec["old_map_sha256"]
    assert canonical(spec["candidate_map"])==spec["candidate_map_sha256"]

class Schedule:
    def __init__(self,mode,old,candidate):
        assert mode in MODES;validate_map(old);validate_map(candidate);assert old!=candidate
        self.mode=mode;self.old=tuple(map(tuple,old));self.candidate=tuple(map(tuple,candidate));self.fired=False
    def initial(self):
        return [list(x) for x in (self.candidate if self.mode=="preinstalled" else self.old)]
    def choose(self,current,measured_started):
        current=tuple(map(tuple,current))
        expected=self.candidate if self.mode=="preinstalled" or self.fired else self.old
        assert current==expected,"Unexpected or partially committed map at proposal boundary"
        if self.mode=="charged" and measured_started and not self.fired:
            self.fired=True
            return [list(x) for x in self.candidate],True
        return [list(x) for x in current],False

def verify_weight_records(peers,mapping,reference=None):
    """Check every slot and replica; optional reference proves logical identity after movement."""
    validate_map(mapping)
    assert len(peers)==8 and sorted(p["rank"] for p in peers)==list(range(8))
    seen=set();logical={}
    for peer in peers:
        assert len(peer["weights"])==48*20
        for record in peer["weights"]:
            layer,slot=record["layer"],record["slot"]
            assert 0<=layer<48 and 0<=slot<160 and slot//20==peer["rank"]
            assert (layer,slot) not in seen;seen.add((layer,slot))
            assert record["expert"]==mapping[layer][slot]
            digests=tuple(record["sha256"])
            assert len(digests)==2 and all(len(d)==64 and not set(d)-set("0123456789abcdef") for d in digests)
            key=(layer,record["expert"])
            assert logical.setdefault(key,digests)==digests,"Replica weights differ"
            if reference is not None:assert reference[key]==digests,"Logical expert changed during transition"
    assert len(seen)==48*160 and len(logical)==48*128
    return logical


def native_hash_reference(ids,mapping):
    assert len(mapping)==160 and set(mapping)==set(range(128))
    result=[]
    for row_index,row in enumerate(ids):
        mapped=[]
        for expert in row:
            assert type(expert) is int and -1<=expert<128
            slots=[i for i,e in enumerate(mapping) if e==expert]
            mapped.append(-1 if expert==-1 else slots[((row_index*2654435769)&0xffffffff)%len(slots)])
        result.append(mapped)
    return result


def agreed_marker(tokens):
    assert len(tokens)==8
    present={x for x in tokens if x is not None}
    assert len(present)<=1,"Ranks observed different measured-phase markers"
    return len(present)==1 and all(x is not None for x in tokens)
