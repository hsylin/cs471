"""Prebind Triton's compiled launcher for the unchanged frozen GPU selector.

The fixed inference stream and internal tensor storage are setup/pre/post
invariants. Original input checks remain, plus alignment-specialization guards.
GPU imports are lazy; no new GPU algorithm, graph, or external CUDA API is used.
"""
import hashlib
from pathlib import Path

from resident_fastpath.fused import FusedMapper

TRITON_VERSION = '3.7.1'
TRITON_SOURCE_SHA256 = {
    'compiler/compiler.py': '021a45ca590052ab6ca6d850eb0a102abf542165836ad92bfb3bb573218fc5d3',
    'runtime/jit.py': '5fe46fbfac257a2b8645b7a2cc709232273d42dd5e1d55de9770b412a9476d77',
    'backends/nvidia/driver.py': '992ba21e12ee170a904f3a1caca24a78ce3dd66b661087faea5973b995480f63',
}
GPU_SOURCE_SHA256 = 'c0cd944f6f2c9ee0daac8b2f9891e57bec816435b27a8692020fd0f94592c5ae'


def _sha(data):
    return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def _storage(tensor):
    return (tensor.data_ptr(),tuple(tensor.shape),str(tensor.dtype),str(tensor.device))


def _signature_table(mapping):
    return [[repr(key),repr(value)] for key,value in sorted(mapping.items(),key=lambda x:repr(x[0]))]


class PreboundCompiledMapper(FusedMapper):
    """Same policy and GPU binary as JIT path, with bound compiled runners."""
    def __init__(self,torch,physical_to_logical_map,batch_tokens=128,top_k=8):
        super().__init__(torch,physical_to_logical_map,batch_tokens,top_k)
        import triton
        from triton import knobs
        import resident_fastpath.fused_gpu as gpu_module
        if triton.__version__ != TRITON_VERSION:
            raise RuntimeError('Unreviewed Triton version')
        root=Path(triton.__file__).resolve().parent
        actual={name:_sha((root/name).read_bytes()) for name in TRITON_SOURCE_SHA256}
        if actual != TRITON_SOURCE_SHA256 or _sha(Path(gpu_module.__file__).read_bytes()) != GPU_SOURCE_SHA256:
            raise RuntimeError('Compiler/runtime or frozen GPU selector source changed')
        self._knobs=knobs
        self._verify_hooks()
        self._fixed_tensors=(self.real_tokens,self._output_i32,self._output_i64,
                             self.homes,self.logical_counts,self.summary)
        self._fixed_storage=tuple(_storage(t) for t in self._fixed_tensors)
        if any(t.data_ptr()%16 for t in self._fixed_tensors):
            raise RuntimeError('Fixed arguments do not satisfy16-byte specialization')
        self._runners={}
        self._compiled={}
        self._kernel_evidence={}
        for label,dtype,output in [('int32',torch.int32,self._output_i32),
                                   ('int64',torch.int64,self._output_i64)]:
            ids=torch.zeros((128,8),device=self.device,dtype=dtype)
            args=(ids,self.real_tokens,output,self.homes,self.logical_counts,self.summary)
            compiled=self.kernel.warmup(*args,N=1024,TOPK=8,grid=(1,),num_warps=4)
            # This normal JIT call establishes identity of the selected binary
            # and initializes it outside timing. Both paths use the same args.
            baseline=self.kernel[(1,)](*args,N=1024,TOPK=8,num_warps=4)
            if baseline is not compiled:
                raise AssertionError('Normal JIT and warmup selected different compiled objects')
            expected_signature=dict(ids='*i32' if label=='int32' else '*i64',
                real_tokens='*i32',output='*i32' if label=='int32' else '*i64',
                homes='*i32',logical_counts='*i32',summary='*i32',N='constexpr',TOPK='constexpr')
            if compiled.src.signature!=expected_signature or compiled.src.constants!={(6,):1024,(7,):8}:
                raise RuntimeError('Compiled pointer signature or constants differ')
            if (compiled.name!='fused_home_map' or compiled.metadata.num_warps!=4
                or compiled.metadata.num_ctas!=1 or compiled.metadata.target.backend!='cuda'
                or compiled.metadata.target.arch!=90
                or compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size):
                raise RuntimeError('Unsupported compiled launch geometry/scratch behavior')
            runner=compiled[(1,1,1)]
            # CompiledKernel runners take all original arguments, including the
            # constexpr positions. CUDA launcher annotations exclude constants
            # from device parameters, exactly as on the normal JIT path.
            runner(*args,1024,8,stream=self.stream_handle)
            self._runners[dtype]=runner
            self._compiled[dtype]=compiled  # Retain module/binary lifetime.
            self._kernel_evidence[label]=dict(compiled_hash=compiled.hash,
                cubin_sha256=_sha(compiled.asm['cubin']),ptx_sha256=_sha(compiled.asm['ptx']),
                signature=_signature_table(compiled.src.signature),
                constants=_signature_table(compiled.src.constants),
                pointer_specialization_attributes=_signature_table(compiled.src.attrs),
                num_warps=compiled.metadata.num_warps,num_ctas=compiled.metadata.num_ctas,
                shared=compiled.metadata.shared,arch=compiled.metadata.target.arch,
                baseline_jit_same_compiled_object=True)
        torch.cuda.current_stream(self.device).synchronize()
        self.calls=0
        self.last_physical_ids=None
        self.verify_context()

    def _verify_hooks(self):
        # CompiledKernel runner preserves dynamic launch-enter/exit metadata
        # hooks. It does not execute JIT pre-run/cache-stage/global inspections;
        # reject configurations needing those omitted JIT services.
        if self.kernel.pre_run_hooks:
            raise RuntimeError('JIT pre-run hooks are unsupported by prebound launch')
        if self.kernel.used_global_vals:
            raise RuntimeError('Mutable JIT global dependencies are unsupported')
        if (self._knobs.runtime.add_stages_inspection_hook is not None
            or self._knobs.runtime.debug
            or self._knobs.compilation.instrumentation_mode):
            raise RuntimeError('Unsupported JIT debug/instrumentation/stage hook')

    def verify_context(self):
        result=super().verify_context()
        if hasattr(self,'_fixed_storage'):
            if tuple(_storage(t) for t in self._fixed_tensors)!=self._fixed_storage:
                raise ValueError('Fixed compiled argument storage changed')
            self._verify_hooks()
            result.update(fixed_argument_storage_verified=True,hook_contract_verified=True)
        return result

    def __call__(self,logical_ids,real_tokens=None):
        torch=self.torch
        # Exactly the existing mapper's shape/layout/device/dtype checks.
        if (tuple(logical_ids.shape)!=(128,8) or logical_ids.dtype not in (torch.int32,torch.int64)
                or not logical_ids.is_contiguous() or logical_ids.device!=self.device):
            raise ValueError('Fused selector input shape/layout/device/type changed')
        real=self.real_tokens if real_tokens is None else real_tokens
        if real_tokens is not None and (real.numel()!=1 or real.dtype!=torch.int32 or real.device!=self.device):
            raise ValueError('real_tokens must be an int32 scalar on this CUDA device')
        if logical_ids.data_ptr()%16 or real.data_ptr()%16:
            raise ValueError('Input pointer violates compiled16-byte alignment specialization')
        output=self._output_i32 if logical_ids.dtype==torch.int32 else self._output_i64
        self._runners[logical_ids.dtype](logical_ids,real,output,self.homes,
            self.logical_counts,self.summary,1024,8,stream=self.stream_handle)
        self.last_physical_ids=output
        self.calls+=1
        return output

    def compiled_launch_snapshot(self):
        self.verify_context()
        return dict(schema='resident_prebound_compiled_launch_v1',mode='prebound_compiled',
            triton_version=TRITON_VERSION,gpu_algorithm_changed=False,
            input_checks_retained=True,pointer_alignment_guard=True,
            stream_handle=self.stream_handle,device=str(self.device),
            fixed_argument_storage_verified=True,hook_contract_verified=True,
            dynamic_launch_metadata_hooks_preserved=True,jit_prerun_hooks_required_empty=True,
            grid=[1,1,1],constexpr_args=[1024,8],calls=self.calls,
            compiler_source_sha256=TRITON_SOURCE_SHA256,
            gpu_source_sha256=GPU_SOURCE_SHA256,kernels=self._kernel_evidence,
            scope='Fixed B128/top8, exact resident map, sequential constructing CUDA stream; no graph/concurrent calls')

    def metadata(self):
        return {**super().metadata(),'compiled_launch':self.compiled_launch_snapshot()}

    compiled_metadata = compiled_launch_snapshot
