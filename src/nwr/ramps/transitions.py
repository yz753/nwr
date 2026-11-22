from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd

CATEGORIES = ['++', '+-', '-+', '--', 'One-side', '//']
COLORS = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba', "#ffcb3a", '#B5BEC7']
COMBINED_CATEGORIES = ['Ramping', 'One-side', '//']
COMBINED_COLORS = ['#d7191c', '#ffcb3a', '#B5BEC7']
BOUND_CATEGORIES = ['+', '-', '/']
BOUND_COLORS = ['#d7191c', '#2b83ba', '#B5BEC7']

BLOCK_ORDER = [0, 1, 2, 0]
CSV_PATH = Path('/Volumes/INCR-NolanLab/ActiveProjects/Yiming/NWR1/analysis/ramps/ramps_classification.csv')
OUTPUT_DIR = CSV_PATH.parent / 'inter_test_block_ramp_transitions'

SOUND_ON_OFF_DAYS = {
    'M01': {'sound_ON': [22,24,26,28], 'sound_OFF': [23,25,27]},
    'M02': {'sound_ON': [20,22,24,26], 'sound_OFF': [21,23,25]},
}
TRANSITION_TYPES = [('Ramping', '//'), ('Ramping', 'One-side'), ('One-side', '//')]


def plot_daily_proportions(ratios, out):
    out = out / 'daily_proportions'
    out.mkdir(exist_ok=True)
    ratios.to_csv(out / 'transition_proportions.csv', index=False)
    for key, group in ratios.groupby(['experiment', 'mouse', 'session', 'trial_type']):
        experiment, mouse, session, trial_type = key
        all_days = list(range(int(group['day_number'].min()), int(group['day_number'].max()) + 1))
        selections = {'all_days': all_days, **SOUND_ON_OFF_DAYS.get(mouse, {
            'custom_1': all_days[::2], 'custom_2': all_days[1::2],
        })}
        for source, target in TRANSITION_TYPES:
            subset = group.loc[(group['source'] == source) & (group['target'] == target)]
            for selection, days in selections.items():
                fig, ax = plt.subplots(figsize=(8, 5))
                for (left, right), color in zip(zip(BLOCK_ORDER[:-1], BLOCK_ORDER[1:]),
                                                ['#af8dc3', '#7fbf7b', '#4d4d4d']):
                    pair = subset.loc[(subset['from_block'] == left) & (subset['to_block'] == right)]
                    values = pair.set_index('day_number')['proportion'].reindex(days)
                    ax.plot(days, values, marker='o', color=color, label=f'Block {left} → {right}')
                ax.set(xticks=days, xlabel='Day', ylabel='Proportion of source-category units',
                       ylim=(-0.03, 1.03),
                       title=f'{experiment} | {mouse} | {session} | Trial type {trial_type}\n'
                             f'{source} → {target} | {selection}')
                ax.legend(frameon=False)
                fig.tight_layout()
                transition = f'{source}_to_{target}'.replace('//', 'nonramping')
                name = f'{experiment}_{mouse}_{session}_trial_type-{trial_type}_{transition}_{selection}'
                fig.savefig(out / f'{name}.png', dpi=180)
                plt.close(fig)
                print(f'Saved {name}.png', flush=True)
                

def plot_transitions(wide, title, categories=CATEGORIES, colors=COLORS):
    n = len(wide)
    gap, height, width = 0.055, 0.72, 0.025
    xs = [0, 0.5, 1, 1.5]
    bottoms, counts = {}, {}
    color_by_category = dict(zip(categories, colors))
    
    for block in (0, 1, 2):
        top = 1.0
        for category in categories:
            count = int((wide[block] == category).sum())
            h = height * count / n
            bottoms[block, category] = top - h
            counts[block, category] = count
            top -= h + gap

    fig, ax = plt.subplots(figsize=(17, 8))
    for column, (left, right) in enumerate(zip(BLOCK_ORDER[:-1], BLOCK_ORDER[1:])):
        flows = pd.crosstab(wide[left], wide[right]).reindex(
            index=categories, columns=categories, fill_value=0
        )
        source = {c: bottoms[left, c] for c in categories}
        target = {c: bottoms[right, c] for c in categories}

        # The category rectangles are arranged from top to bottom, while the
        # flow coordinates accumulate from bottom to top. Reverse both loops
        # so flow bands retain the same vertical ordering at each endpoint.
        for a in reversed(categories):
            for b in reversed(categories):
                count = int(flows.loc[a, b])
                if not count:
                    continue
                h = height * count / n
                y0, y1 = source[a], target[b]
                x0, x1 = xs[column] + width / 2, xs[column + 1] - width / 2
                dx = (x1 - x0) * 0.45
                vertices = [(x0, y0), (x0+dx, y0), (x1-dx, y1), (x1, y1),
                            (x1, y1+h), (x1-dx, y1+h), (x0+dx, y0+h),
                            (x0, y0+h), (x0, y0)]
                codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                         MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                         MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(vertices, codes), facecolor=color_by_category[a],
                                       edgecolor='none', alpha=0.38))
                source[a] += h
                target[b] += h
        assert int(flows.to_numpy().sum()) == n

    for block, x in zip(BLOCK_ORDER, xs):
        ax.text(x, 1.09, f'Test block {block}', ha='center', fontsize=17, color='black')
        for category, color in zip(categories, colors):
            count = counts[block, category]
            h = height * count / n
            y = bottoms[block, category]
            ax.add_patch(Rectangle((x-width/2, y), width, h,
                                   facecolor=color, edgecolor='white', linewidth=0.5))
            ax.text(x + 0.023, y + h/2, f'{category}  ({count})',
                    va='center', fontsize=11, color='black',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
    ax.set(xlim=(-0.06, 1.74), ylim=(-0.06, 1.16))
    ax.axis('off')
    fig.suptitle(title, fontsize=19, color='black', y=0.97)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.87, bottom=0.12)
    return fig


    # if ratios:
    #     plot_daily_proportions(pd.DataFrame(ratios), out)


