# Allow direct execution with the existing local/Eddie environment.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nwr.ramps.transitions import (
    BLOCK_ORDER,
    BOUND_CATEGORIES,
    BOUND_COLORS,
    CATEGORIES,
    COLORS,
    COMBINED_CATEGORIES,
    COMBINED_COLORS,
    CSV_PATH,
    MPath,
    OUTPUT_DIR,
    Path,
    PathPatch,
    Rectangle,
    SOUND_ON_OFF_DAYS,
    TRANSITION_TYPES,
    matplotlib,
    np,
    pd,
    plot_daily_proportions,
    plot_transitions,
    plt,
)

def main():
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH, keep_default_na=False)
    group_cols = ['experiment', 'mouse', 'day', 'session', 'trial_type']
    for col in ['unit_id', 'trial_type', 'test_block']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    df = df.loc[df['test_block'].isin([0, 1, 2])].copy()
    # Do not group unknown/NA classifications into Other.
    mapping = {c: c for c in ['++', '+-', '-+', '--', '//']}
    mapping.update({
        c: 'One-side'
        for c in ['+/', '-/', '/+', '/-']
    })
    df['category'] = df['ramp_type'].astype(str).str.strip().map(mapping)
    keys = group_cols + ['unit_id', 'test_block']
    if df.duplicated(keys).any():
        raise ValueError('Duplicate unit/block rows: select one analysis run/NWB per session first.\n'
                         + df.loc[df.duplicated(keys, keep=False), keys].head(10).to_string(index=False))
    reports, transitions, ratios = [], [], []
    for key, group in df.groupby(group_cols, sort=True, dropna=False):
        wide = group.pivot(index='unit_id', columns='test_block', values='category')
        wide = wide.reindex(columns=[0, 1, 2])
        included = wide.dropna()
        wide_raw = group.pivot(
            index='unit_id',
            columns='test_block',
            values='ramp_type',
        ).reindex(columns=[0, 1, 2])

        wide_raw = wide_raw.apply(
            lambda column: column.astype(str).str.strip()
        )
        included_raw = wide_raw.loc[included.index]
        metadata = dict(zip(group_cols, key))
        reports.append(dict(metadata, included=len(included)))
        combined = included.replace({c: 'Ramping' for c in ['++', '+-', '-+', '--']})
        for left, right in zip(BLOCK_ORDER[:-1], BLOCK_ORDER[1:]):
            for source, target in TRANSITION_TYPES:
                source_mask = combined[left] == source
                numerator = int((source_mask & (combined[right] == target)).sum())
                denominator = int(source_mask.sum())
                ratios.append(dict(metadata, day_number=int(str(key[2]).lstrip('Dd')),
                                   from_block=left, to_block=right, source=source, target=target,
                                   numerator=numerator, denominator=denominator,
                                   proportion=numerator / denominator if denominator else np.nan))
        if included.empty:
            print(f'Skipping {key}: no complete valid units', flush=True)
            continue
        for left, right in zip(BLOCK_ORDER[:-1], BLOCK_ORDER[1:]):
            for (a, b), count in included.groupby([left, right]).size().items():
                transitions.append(dict(metadata, from_block=left, to_block=right,
                                        from_type=a, to_type=b, count=int(count)))
        name = '_'.join(str(v) for v in key)
        title = f'{key[0]} | {key[1]} | {key[2]} | {key[3]} | Trial type {key[4]}'
        all_ramps_combined = included.replace({
            '++': 'Ramping',
            '+-': 'Ramping',
            '-+': 'Ramping',
            '--': 'Ramping',
        })

        # First sign is outbound; second sign is homebound.
        outbound = included_raw.apply(lambda column: column.str[0])
        homebound = included_raw.apply(lambda column: column.str[1])

        variants = [
            (
                'separate',
                included,
                CATEGORIES,
                COLORS,
                'Full ramp classification',
            ),
            (
                'combined',
                all_ramps_combined,
                COMBINED_CATEGORIES,
                COMBINED_COLORS,
                'All two-sided ramps combined',
            ),
            (
                'outbound_separate',
                outbound,
                BOUND_CATEGORIES,
                BOUND_COLORS,
                'Outbound classification',
            ),
            (
                'homebound_separate',
                homebound,
                BOUND_CATEGORIES,
                BOUND_COLORS,
                'Homebound classification',
            ),
        ]

        for suffix, plot_data, categories, colors, subtitle in variants:
            fig = plot_transitions(
                plot_data,
                f'{title}\n{subtitle}',
                categories=categories,
                colors=colors,
            )
            output = out / f'{name}_{suffix}.png'
            fig.savefig(output, dpi=180)
            plt.close(fig)
            print(f'Saved {output}', flush=True)
    pd.DataFrame(reports).to_csv(out / 'included_units.csv', index=False)
    pd.DataFrame(transitions).to_csv(out / 'transition_counts.csv', index=False)

if __name__ == '__main__':
    main()
