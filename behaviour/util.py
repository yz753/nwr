class Mouse_Info:
    mouse_colour_palette_dict = {
        'M1': '#b2df8a',
        'M2': '#1f78b4',
        'M3': '#a6cee3',
        'M4': '#33a02c',
        'M5': '#fb9a99',
        'M6': '#e31a1c',
        'M7': '#fdbf6f',
        'M8': '#ff7f00',
        'M9': '#cab2d6',
        'M10': '#6a3d9a',
        'M11': '#ffff99',
        'M12': '#b15928'
    }
    
    trial_type_colour_palette_dict = {
        0: 'blue', # b
        1: 'black', # nb
        'b': 'blue',
        'nb': 'black',
    }
    
    test_days_dict = {
        'M1': {
            'sound_ON': [22,24,26,28,30],
            'sound_OFF': [23,25,27,29,31],
        },
        'M2': {
            'sound_ON': [20,22,24,26,28],
            'sound_OFF': [21,23,25,27,29],
        }
    } # fill in the dict only with test days