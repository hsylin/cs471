import copy
import unittest
import numpy as np

import audit_residual_development as audit
import audit_residual_records as fresh
import residual_preflight_primitives as preflight


def kernel(dtype):
    signature=dict(ids='*i32' if dtype=='int32' else '*i64',real_tokens='*i32',
        output='*i32' if dtype=='int32' else '*i64',homes='*i32',logical_counts='*i32',summary='*i32',N='constexpr',TOPK='constexpr')
    return dict(signature=[[repr(k),repr(v)] for k,v in sorted(signature.items(),key=lambda x:repr(x[0]))],
        constants=[['(6,)','1024'],['(7,)','8']],baseline_jit_same_compiled_object=True,
        num_warps=4,num_ctas=1,arch=90,shared=512,pointer_specialization_attributes=[['(0,)','16']],
        compiled_hash='1'*64,cubin_sha256='2'*64,ptx_sha256='3'*64)


def row():
    peers=[]
    for rank in range(2):
        peer={}
        for scope in ('start','end','audit'):
            meta=dict(schema='resident_prebound_compiled_launch_v1',mode='prebound_compiled',
                triton_version='3.7.1',gpu_algorithm_changed=False,input_checks_retained=True,pointer_alignment_guard=True,
                device=f'cuda:{rank}',fixed_argument_storage_verified=True,hook_contract_verified=True,
                dynamic_launch_metadata_hooks_preserved=True,jit_prerun_hooks_required_empty=True,
                grid=[1,1,1],constexpr_args=[1024,8],calls=0 if scope=='start' else 512,
                compiler_source_sha256=audit.COMPILER_PINS.copy(),gpu_source_sha256=audit.GPU_SOURCE,
                stream_handle=rank+10,kernels={dtype:kernel(dtype) for dtype in ('int32','int64')})
            peer[scope]=dict(compiled_launch=meta,fused_stream_context=dict(stream_handle=rank+10))
        peers.append(peer)
    return dict(variant='fix',per_rank_state=peers)


class CompiledAuditTests(unittest.TestCase):
    def test_valid(self):self.assertEqual(set(audit.compiled_check(row())),{'int32','int64'})
    def test_mutations(self):
        mutations=[('calls',511),('gpu_algorithm_changed',True),('input_checks_retained',False),
            ('pointer_alignment_guard',False),('stream_handle',999),('compiler_source_sha256',{})]
        for name,value in mutations:
            with self.subTest(name=name):
                r=row();r['per_rank_state'][1]['end']['compiled_launch'][name]=value
                with self.assertRaises(ValueError):audit.compiled_check(r)
    def test_wrong_constants_and_crossrank_binary(self):
        for field,value in [('constants',[['(6,)','512'],['(7,)','8']]),('cubin_sha256','4'*64)]:
            r=row();r['per_rank_state'][1]['audit']['compiled_launch']['kernels']['int32'][field]=value
            with self.assertRaises(ValueError):audit.compiled_check(r)
    def test_no_compiled_metadata_in_control(self):
        r=row();r['variant']='previous'
        with self.assertRaises(ValueError):audit.compiled_check(r)


class GeometryAuditTests(unittest.TestCase):
    def test_actual_grid_and_padding(self):
        ids=np.tile(np.arange(8),(1,128,1));extent=9088
        record=dict(physical_ids_sha256=audit.base.integer_hash(ids[0]),
            independent_local_counts=dict(tokens=1024,active=8,padded_rows=1024,capacity_bound=5056),
            actual_padded_rows=1024,actual_m_tiles=16,backing_pointer_preserved=True,valid_prefix_equal=True,
            token_conservation_verified=True,scope='actual_native_gemm_pair',
            gemms=[dict(n=n,top_k=k,sorted_extent=extent,m_programs=142,grid_programs=142*(n//128)) for n,k in [(1536,8),(2048,1)]])
        proof=dict(probe_only=True,actual_padding_verified=True,alignment_calls=1,probe_records=[record],
            enabled=False,config=audit.base.CONFIG,native_allocation_preserved=True,native_alignment_algorithm_preserved=True)
        self.assertEqual(audit.geometry_proof(proof,ids,0,'stockoff'),1)
        for mutation in ('padding','grid','prefix'):
            wrong=copy.deepcopy(proof);r=wrong['probe_records'][0]
            if mutation=='padding':r['actual_padded_rows']+=64
            if mutation=='grid':r['gemms'][0]['grid_programs']-=12
            if mutation=='prefix':r['valid_prefix_equal']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):audit.geometry_proof(wrong,ids,0,'stockoff')


class FreshContractTests(unittest.TestCase):
    def plan(self):
        p=dict(phase='pilot',banks=[dict(block_id=g,dataset=d) for g,d in
            [('g0','gsm8k_train'),('g1','gsm8k_train'),('w0','wildchat'),('w1','wildchat')]],cases=[])
        order={'g0':['stockoff','previous','fix'],'g1':['stockoff','fix','previous'],
               'w0':['stockoff','fix','previous'],'w1':['previous','stockoff','fix']}
        for g,o in order.items():
            for rho in (.5,.85):
                for t,permutation in [('code',o),('conversation',o[::-1])]:
                    p['cases'].append(dict(block_id=g,arrival_type=t,rho=rho,arm_order=permutation))
        return p
    def test_order_reversal_and_near_balance(self):fresh.order_check(self.plan())
    def test_wrong_order_and_missing_case(self):
        for mutation in ('reverse','missing'):
            p=self.plan()
            if mutation=='reverse':p['cases'][1]['arm_order']=p['cases'][0]['arm_order']
            else:p['cases'].pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):fresh.order_check(p)
    def test_faster_independent_oracle_matches_scalar_spec(self):
        rng=np.random.default_rng(917)
        for _ in range(8):
            ids=np.stack([rng.choice(128,8,replace=False) for row in range(128)])
            physical,counts=fresh.oracle(ids)
            np.testing.assert_array_equal(physical,audit.base.physical_oracle(ids))
            self.assertEqual(int(counts['physical_counts'].sum()),1024)
    def test_partial_and_zero_probe_masks(self):
        for name,ids,real in preflight.cases():
            result=preflight.expected(ids,real)
            self.assertEqual(int(result['logical_counts'].sum()),real*8)
            self.assertTrue(np.all(result['physical_ids'][real:]==-1))
            self.assertLessEqual(max(result['summary'][-2:]),64)


if __name__=='__main__':unittest.main()
