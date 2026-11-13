"""Pure per-layer B3/B4 admission rule from the supplied SkewBench proposal.

Counts are authentic logical assignments, converted using the map at observation
(time of routing), before any async commit changes that map. Equal replica sharing
is a prediction only. Rejection returns the current map to native async execution.
"""
from __future__ import annotations
import numpy as np


def decide(previous, recent, current, candidate, *, ranks=8, gain_threshold=.05,
           stability_threshold=.05, require_stability=True, recent_complete=True,
           previous_complete=True):
    old,new=np.asarray(current,dtype=np.int64),np.asarray(candidate,dtype=np.int64)
    a,b=np.asarray(previous,dtype=np.float64),np.asarray(recent,dtype=np.float64)
    if (a.shape!=b.shape or a.ndim!=1 or old.shape!=new.shape or old.ndim!=1
        or old.size%ranks or np.any(a<0) or np.any(b<0) or not np.isfinite(a).all()
        or not np.isfinite(b).all() or np.any(old<0) or np.any(new<0)
        or np.any(old>=len(b)) or np.any(new>=len(b))
        or not 0<gain_threshold<1 or not 0<=stability_threshold<=1):
        raise ValueError('Invalid counts, mapping or thresholds')
    if np.any(np.bincount(old,minlength=len(b))==0) or np.any(np.bincount(new,minlength=len(b))==0):
        raise ValueError('Placement must cover every logical expert')
    at,bt=float(a.sum()),float(b.sum())
    tv=float(np.abs(a/at-b/bt).sum()/2) if at and bt else None
    def peak(m):
        copies=np.bincount(m,minlength=len(b))
        return float((b[m]/copies[m]).reshape(ranks,-1).sum(axis=1).max())
    before,after=peak(old),peak(new)
    gain=(before-after)/before if before else None
    reason=('incomplete_recent' if not recent_complete else 'zero_recent' if not bt
            else 'incomplete_previous' if require_stability and not previous_complete
            else 'zero_previous' if require_stability and not at else 'unchanged' if np.array_equal(old,new)
            else 'insufficient_gain' if gain is None or gain<gain_threshold
            else 'unstable' if require_stability and (tv is None or tv>stability_threshold) else 'allow')
    return {'allow':reason=='allow','reason':reason,'tv':tv,'predicted_gain':gain,
            'current_peak_assignments':before,'candidate_peak_assignments':after,
            'previous_assignments':at,'recent_assignments':bt,
            'changed_physical_slots':int(np.count_nonzero(old!=new))}
