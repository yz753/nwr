import numpy as np
import pynapple as nap

NWB_ADAPTER_READY = True

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


def get_ramps_input(nwbfile, unit_id) -> dict:
    # only test days nwb files have keys: block_idx
    if 'block_idx' in nwbfile.keys():
      training_or_test = 'test'
      # generate test day block idx IntervalSet (0: rewarded, 1: non-rewarded, 2: re-rewarded)
      thresholding_0_1_2 = [
          (None, 0.5),
          (0.5, 1.5),
          (1.5, None),
      ]
      test_block_intervalset = get_intervalset(nwbfile['block_idx'], thresholding_0_1_2)

    else:
      training_or_test = 'training'
      test_block_intervalset = None
      
    # compute trial type IntervalSet (0: b, 1: nb)
    thresholding_0_1 = [
        (None, 0.5),
        (0.5, None),
    ]
    trial_type_intervalset = get_intervalset(nwbfile['trial_type'], thresholding_0_1)
    
    return {
      'training_or_test': training_or_test, # str
      'P': nwbfile['P'], # Tsd
      'trial_idx': (nwbfile['trial_number']-1), # Tsd
      'moving': nwbfile['moving'], # IntervalSet
      'trials': trial_type_intervalset, # IntervalSet
      'test_block': test_block_intervalset, # IntervalSet
      'cluster': nap.TsGroup({
        int(unit_id): nwbfile['units'][unit_id]
        }) # TsGroup
    }
    
