"""Explicit static-resident treatments; stockoff remains the unmodified facade."""
from resident_fastpath.native import StaticResidentLayer
from stock_comparison.native import StockLayer
from resident_residual.alignment import trim_native_alignment

class ResidualLayer(StaticResidentLayer):
    def __init__(self,rt,weights,arm,*,alignment_trim=False,mapper_input_checks=True,selector_launch='jit'):
        super().__init__(rt,weights,selector='fused')
        if arm!='fix':rt.layers[arm]=rt.layers.pop('fix')
        self.arm=arm;self.spec={**self.spec,'arm':arm}
        self.selector_launch=selector_launch
        if selector_launch=='prebound_compiled':
            from resident_residual.compiled import PreboundCompiledMapper
            self.mapper=PreboundCompiledMapper(self.torch,self.initial_map,128)
            self.mapping_kernel=self.mapper
            self.mapper.verify_metadata(self.layer_state.logical_to_physical_map,self.layer_state.logical_replica_count,self.ms.physical_to_logical_map[0])
        elif selector_launch!='jit':raise ValueError('Unknown launcher')
        self.alignment_trim=bool(alignment_trim)
        self.mapper_input_checks=bool(mapper_input_checks)
        if not self.mapper_input_checks:raise ValueError('Unchecked mapper is diagnostic-only; no selected implementation')
    def snapshot(self):
        result=super().snapshot()
        result.update(alignment_trim=getattr(self,'alignment_trim',False),
                      mapper_input_checks=getattr(self,'mapper_input_checks',True),
                      selector_launch=getattr(self,'selector_launch','jit'),
                      compiled_launch=self.mapper.compiled_launch_snapshot() if getattr(self,'selector_launch','jit')=='prebound_compiled' else None)
        return result

def alignment_scope(layer,*,probe=False):
    return trim_native_alignment(layer,enabled=getattr(layer,'alignment_trim',False),probe=probe,policy='one_copy_home')

def build_layers(rt,weights,*,selected):
    off=StockLayer(rt,weights,'stockoff')
    previous=ResidualLayer(rt,weights,'previous')
    fixed=ResidualLayer(rt,weights,'fix',alignment_trim=selected['alignment_trim'],mapper_input_checks=selected['mapper_input_checks'],selector_launch=selected['selector_launch'])
    return {'stockoff':off,'previous':previous,'fix':fixed}
