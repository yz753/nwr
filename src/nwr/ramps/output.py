import matplotlib.pyplot as plt
import numpy as np


def store_results(
    results_null,
    experiment,
    mouse_id,
    day,
    cluster_id,
    output_dir,
    reward_position,
    homebound,
    outbound,
    trial_type,
    training_or_test,
    test_block
    ):
    '''plot tc and save numbers in npz'''
    # plotting
    tc = results_null['tc']
    # shade reward zone & black boxes
    plt.axvspan(
        int(reward_position[0]+2),
        reward_position[1],
        color='lightgreen', alpha=0.3
        )
    plt.axvspan(0, 30, color='grey', alpha=0.3)
    plt.axvspan(
        homebound[1],
        int(homebound[1]+30),
        color='grey', alpha=0.3
        )
    # best fit line
    outbound_x = np.linspace(outbound[0], outbound[1], 150)
    homebound_x = np.linspace(homebound[0], homebound[1], 150)
    plt.plot(
        outbound_x,
        results_null["outbound_slope"]*outbound_x +
        results_null["outbound_intercept"],
        color='green'
        )
    plt.plot(
        outbound_x,
        np.mean(
            results_null['null']["outbound_slope"]
            )
        *outbound_x +
        results_null["outbound_intercept"],
        color='gray'
        )
    plt.plot(
        homebound_x,
        results_null["homebound_slope"]*homebound_x +
        results_null["homebound_intercept"],
        color='green'
        )
    plt.plot(
        homebound_x,
        np.mean(
            results_null['null']["homebound_slope"]
            )
        *homebound_x +
        results_null["homebound_intercept"],
        color='gray'
        )
    # tc
    plt.plot(tc.coords['0'].values, tc.values, color='blue')
      
    plt.xlabel('Position (cm)')
    plt.ylabel('Firing rate (Hz)')
    plt.title(
        f'{experiment} {training_or_test} {mouse_id} {day} unit {cluster_id} trial_type={trial_type[0]} test_block={test_block[0] if test_block is not None else None}'
        )
    plt.savefig(
        f'{output_dir}/{experiment}_{training_or_test}_{mouse_id}_{day}_unit-{cluster_id}_trial_type-{trial_type[0]}_test_block-{test_block[0] if test_block is not None else None}.png',
        dpi=80
        )
    plt.close()
    
    # save npz
    np.savez_compressed(
        output_dir / f'{experiment}_{training_or_test}_{mouse_id}_{day}_unit-{cluster_id}_trial_type-{trial_type[0]}_test_block-{test_block[0] if test_block is not None else None}.npz',
        tc = tc.values,
        location = tc.coords['0'].values.tolist(),
        unit = int(tc.coords['unit'].values),
        occupancy = tc.attrs['occupancy'],
        bin_edges = tc.attrs['bin_edges'],
        fs = float(tc.attrs['fs']),
        rates = float(tc.attrs['rates'][0]),
    )


def make_csv(results_null):
    return {
        '_ramps_smooth_sigma': results_null.get('_smooth_sigma'),
        'outbound_slope': results_null.get('outbound_slope'),
        'outbound_intercept': results_null.get('outbound_intercept'),
        'outbound_pval': results_null.get('outbound_pval'),
        'outbound_region': results_null.get('outbound_region'),
        'homebound_slope': results_null.get('homebound_slope'),
        'homebound_intercept': results_null.get('homebound_intercept'),
        'homebound_pval': results_null.get('homebound_pval'),
        'homebound_region': results_null.get('homebound_region'),
        'outbound_sig': results_null.get('outbound_sig'),
        'outbound_sign': results_null.get('outbound_sign'),
        'homebound_sig': results_null.get('homebound_sig'),
        'homebound_sign': results_null.get('homebound_sign'),
        'outbound_valid_bins': results_null.get('outbound_valid_bins'),
        'homebound_valid_bins': results_null.get('homebound_valid_bins'),
    }
