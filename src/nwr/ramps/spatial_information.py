def make_csv(results_null):
    return {
        'spatial_information': results_null.get('spatial_information'),
        '_si_smooth_sigma': results_null.get('_smooth_sigma'),
        'si_sig': results_null.get('sig'),
        'si_pval': results_null.get('pval'),
    }