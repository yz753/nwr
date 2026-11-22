import numpy as np
import pynapple as nap

def get_intervalset(tsd, thresholds):
    '''thresholds: to define 0,1,2, use None, 0.5, 1.5, None; to define 0,1, use None, 0.5, None'''
    intervals = []
    for low, high in thresholds:
        ts = tsd
        if low is not None:
            ts = ts.threshold(low, method='above')
        if high is not None:
            ts = ts.threshold(high, method='below')
        intervals.append(ts.time_support)
        
    starts = np.concatenate([iv.start for iv in intervals])
    ends = np.concatenate([iv.end for iv in intervals])
    labels = np.concatenate([
        np.full(len(iv), i, dtype=int)
        for i, iv in enumerate(intervals)
        ])
    
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]
    labels = labels[order]
    return nap.IntervalSet(
        start=starts,
        end=ends,
        metadata={'type': labels}
        )

