import unittest
from full_serving.boundary_gated_capture import slice_events
class SliceCorrelation(unittest.TestCase):
    def test_duplicate_host_correlation_stays_unknown(self):
        def e(cat,name,ts,dur,pid,args):return dict(cat=cat,name=name,ph='X',ts=ts,dur=dur,pid=pid,args=args)
        events=[e('user_annotation','start',0,1,1,{}),e('user_annotation','end',10,1,1,{})]
        events += [e('cuda_runtime','launch',2,1,p,{'correlation':c}) for p,c in [(1,5),(2,5),(1,6),(1,6)]]
        events += [e('kernel','k',20,1,9,{'correlation':c}) for c in [5,6,7]]
        selected,_=slice_events(events,'start','end')
        self.assertEqual([x['args']['correlation'] for x in selected if x['cat']=='kernel'],[6])
    def test_overlapping_unassigned_activity_preserved(self):
        events=[dict(cat='user_annotation',name=n,ph='X',ts=t,dur=1,pid=1,args={}) for n,t in [('start',0),('end',10)]]
        events.append(dict(cat='kernel',name='unknown',ph='X',ts=3,dur=20,pid=9,args={'correlation':999}))
        selected,_=slice_events(events,'start','end');self.assertTrue(any(x['name']=='unknown' for x in selected))
