"""Read-only CUPTI stream identity queries at diagnostic capture boundaries."""
import ctypes,hashlib
from pathlib import Path

class StreamIdentity:
 def __init__(self):
  paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libcupti.so' in line and line.split()[-1].startswith('/')}
  assert len(paths)==1,('Expected one profiler-loaded CUPTI library',paths)
  self.path=next(iter(paths));self.cupti=ctypes.CDLL(self.path);self.cuda=ctypes.CDLL('libcuda.so.1')
  self.cuda.cuCtxGetCurrent.argtypes=[ctypes.POINTER(ctypes.c_void_p)];self.cuda.cuCtxGetCurrent.restype=ctypes.c_int
  self.cuda.cuCtxGetDevice.argtypes=[ctypes.POINTER(ctypes.c_int)];self.cuda.cuCtxGetDevice.restype=ctypes.c_int
  self.cupti.cuptiGetContextId.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)];self.cupti.cuptiGetContextId.restype=ctypes.c_int
  self.cupti.cuptiGetStreamIdEx.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint8,ctypes.POINTER(ctypes.c_uint32)];self.cupti.cuptiGetStreamIdEx.restype=ctypes.c_int
 def resolve(self,device,handle):
  ctx=ctypes.c_void_p();dev=ctypes.c_int();cid=ctypes.c_uint32();sid=ctypes.c_uint32()
  a=self.cuda.cuCtxGetCurrent(ctypes.byref(ctx));b=self.cuda.cuCtxGetDevice(ctypes.byref(dev))
  base=dict(stream_device=device,stream_handle=handle,method='cuptiGetStreamIdEx',library_path=self.path)
  if a or b or not ctx.value or dev.value!=device:return dict(base,status='unknown',reason='current_context_device_mismatch',return_codes=[a,b])
  c=self.cupti.cuptiGetContextId(ctx,ctypes.byref(cid))
  # Concrete non-null stream handles only: do not guess legacy/per-thread defaults.
  if handle in (0,1,2):return dict(base,status='unknown',reason='special_stream_semantics_unresolved')
  d=self.cupti.cuptiGetStreamIdEx(ctx,ctypes.c_void_p(handle),0,ctypes.byref(sid))
  if c or d:return dict(base,status='unknown',reason='CUPTI_query_failed',return_codes=[c,d])
  return dict(base,status='resolved',device=dev.value,context=cid.value,stream=sid.value)
