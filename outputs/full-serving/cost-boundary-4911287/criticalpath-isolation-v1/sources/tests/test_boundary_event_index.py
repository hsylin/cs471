import unittest,random,collections
from full_serving.boundary_cuda_event_analysis import predecessor_index,predecessor_at
class IndexedPredecessor(unittest.TestCase):
    def test_matches_scan_with_duplicate_correlations_and_ties(self):
        rng=random.Random(47);events=[];apis=collections.defaultdict(list)
        for i in range(300):
            corr=i//2 if i<20 else i
            e=dict(ts=rng.randrange(30),dur=rng.randrange(10),args=dict(device=0,context=1,stream=i%3,correlation=corr))
            events.append(e);apis[corr].append(dict(ts=rng.randrange(30),dur=rng.randrange(5)))
        idx=predecessor_index(events,apis)
        for stream in range(4):
            source=(0,1,stream)
            for t in range(-1,40):
                candidates=[e for e in events if e['args']['stream']==stream and len(apis[e['args']['correlation']])==1 and apis[e['args']['correlation']][0]['ts']+apis[e['args']['correlation']][0]['dur']<=t]
                expected=max(candidates,key=lambda e:e['ts']+e['dur']) if candidates else None
                self.assertIs(predecessor_at(idx,source,t),expected)
    def test_zero_duration_and_equal_launch_endpoint(self):
        e=dict(ts=9,dur=0,args=dict(device=0,context=1,stream=2,correlation=3))
        idx=predecessor_index([e],{3:[dict(ts=4,dur=1)]})
        self.assertIsNone(predecessor_at(idx,(0,1,2),4.9));self.assertIs(predecessor_at(idx,(0,1,2),5),e)
