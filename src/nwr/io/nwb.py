from nwr.behaviour.intervals import get_intervalset
import numpy as np
import pynapple as nap

NWB_ADAPTER_READY = True



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
    
