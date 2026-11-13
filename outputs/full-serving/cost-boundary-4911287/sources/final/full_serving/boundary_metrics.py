"""Five-second diagnostic sampler of allowlisted engine gauges only."""
import contextlib,json,math,re,threading,time,urllib.request
from pathlib import Path
NAMES={"num_requests_running","num_requests_waiting","kv_cache_usage_perc","num_preemptions_total"}
def parse(text):
 result=[]
 for line in text.splitlines():
  m=re.fullmatch(r"vllm:([a-z_]+)\{([^}]*)\}\s+(\S+)",line)
  if not m or m[1] not in NAMES:continue
  engine=re.search(r'(?:^|,)engine="([0-7])"(?:,|$)',m[2])
  if engine is None:continue
  value=float(m[3]);assert math.isfinite(value) and value>=0
  if m[1]=="kv_cache_usage_perc":assert value<=1
  result.append(dict(metric=m[1],engine=int(engine[1]),value=value))
 return result
@contextlib.contextmanager
def sample(url,output):
 stop=threading.Event();output=Path(output)
 def run():
  while not stop.is_set():
   record=dict(start_wall_ns=time.time_ns(),period_s=5)
   try:
    with urllib.request.urlopen(url+"/metrics",timeout=10) as response:
     record["gauges"]=parse(response.read().decode())
    record["status"]="observed"
   except Exception as exc:record.update(status="unobserved",error_type=type(exc).__name__)
   record["end_wall_ns"]=time.time_ns()
   with output.open("a") as f:f.write(json.dumps(record,sort_keys=True)+"\n")
   stop.wait(5)
 thread=threading.Thread(target=run,daemon=True);thread.start()
 try:yield
 finally:
  stop.set();thread.join(timeout=12)
  assert not thread.is_alive(),"Diagnostic metric sampler failed to stop"
