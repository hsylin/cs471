import unittest
from full_serving.boundary_critical_exposure import ActivityIndex,api_scopes,family,sidecar_family
class ExposureTests(unittest.TestCase):
    def test_overlap_includes_copy_and_multiple_streams(self):
        idx=ActivityIndex([(0,5),(4,9),(11,12)])
        self.assertEqual(idx.trailing_unobserved(1,8),0)
        self.assertEqual(idx.trailing_unobserved(1,10),1)
        self.assertEqual(idx.trailing_unobserved(10,15),3)
    def test_empty_observation_is_not_speedup(self):self.assertEqual(ActivityIndex([]).trailing_unobserved(5,10),5)
    def test_prior_activity_before_scope(self):self.assertEqual(ActivityIndex([(0,1)]).trailing_unobserved(5,10),5)
    def test_future_activity_does_not_mask_tail(self):self.assertEqual(ActivityIndex([(20,30)]).trailing_unobserved(5,10),5)
    def test_only_same_thread_cpu_annotations(self):
        events=[dict(cat=c,ph='X',name='cs471::EPLB.step',tid=t,ts=0,dur=20) for c,t in [('gpu_user_annotation',1),('user_annotation',2)]]
        events.append(dict(cat='cuda_runtime',ph='X',name='launch',tid=1,ts=1,dur=2,args={'correlation':7}))
        self.assertEqual(api_scopes(events)[7]['scopes'],[])
    def test_nested_scope_and_driver_api(self):
        es=[dict(cat='user_annotation',ph='X',name='cs471::'+n,tid=1,ts=a,dur=b-a) for n,a,b in [('execute',0,20),('EPLB.prepare_forward',2,10)]]
        es.append(dict(cat='cuda_driver',ph='X',name='launch',tid=1,ts=3,dur=2,args={'correlation':9}))
        self.assertEqual(family(api_scopes(es)[9]['scopes']),'EPLB_main')
    def test_sidecar_requires_definite_containment(self):
        a=dict(thread=2,api_start=5,api_end=6);s=[dict(thread=2,name='worker.transfer_layer',start_wall_ns=10000,end_wall_ns=20000)]
        self.assertEqual(sidecar_family(a,s,6,7),'worker')
        self.assertIsNone(sidecar_family(a,s,0,7))
    def test_same_name_other_thread_rejected(self):
        a=dict(thread=1,api_start=5,api_end=6);s=[dict(thread=2,name='worker.transfer_layer',start_wall_ns=10000,end_wall_ns=20000)]
        self.assertIsNone(sidecar_family(a,s,6,7))
if __name__=='__main__':unittest.main()
