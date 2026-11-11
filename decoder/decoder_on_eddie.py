#!/usr/bin/env python3
"""Submit one decoder array task per VR session; errors go to SGE logs."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path('/exports/cmvm/datastore/sbms/groups/INCR-NolanLab/ActiveProjects/Yiming/NWR1/processed')
MANIFEST = Path('/exports/eddie/scratch/s2155699/ephys/nwr/nwb_units.jsonl')

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PATHS import INPUT_ROOT


def read_tasks(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--mice', nargs='+', help='e.g. M1 M2 or M1,M2 or 1,2')
    parser.add_argument('--days', nargs='+', help='e.g. D2 D3 or D2,D3 or 2,3')
    parser.add_argument('--output-dir', type=Path, default=Path(os.environ.get(
        'DECODER_OUTPUT_DIR', '/exports/eddie/scratch/s2155699/ephys/decoder_results')))
    parser.add_argument('--loss-fn', choices=['wasserstein', 'cross_entropy'],
                        default=os.environ.get('DECODER_LOSS_FN', 'wasserstein'))
    parser.add_argument('--phase', choices=['stage', 'fit'], help=argparse.SUPPRESS)
    parser.add_argument('--tasks', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.loss_fn not in {'wasserstein', 'cross_entropy'}:
        parser.error('Invalid DECODER_LOSS_FN')

    if args.phase:
        tasks = read_tasks(args.tasks)
        ready = args.tasks.parent / 'staging_complete'
        if args.phase == 'stage':
            for task in tasks:
                target = Path(task['nwb_path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([
                    'rsync', '-rt', '--ignore-times',
                    str(SOURCE / task['relative_path']),
                    str(target),
                ], check=True)
            ready.touch()
            return
        if not ready.exists():
            raise RuntimeError('Staging failed; see staging job logs')
        task = tasks[int(os.environ['SGE_TASK_ID']) - 1]
        import torch
        from decode import decode
        from output import save_and_plot
        torch.set_num_threads(1)
        print(f'Starting {task["experiment"]} M{task["mouse"]} D{task["day"]}', flush=True)
        result = decode(
            task, 
            output_dir=args.output_dir, 
            loss_fn=args.loss_fn,
            seed=0, 
            bin_size=0.1, 
            n_position_bins=200, 
            n_epochs=200,
        )
        save_and_plot(result, args.output_dir)
        print(f'Finished.', flush=True)
        return

    if args.all and (args.mice or args.days):
        parser.error('Use --all OR mouse/day filters')
    if not (args.all or args.mice or args.days):
        parser.error('Use --all, --mice and/or --days')
    def ids(values, prefix):
        return {int(v.upper().removeprefix(prefix))
                for value in values for v in value.split(',')} if values else None
    mice, days = ids(args.mice, 'M'), ids(args.days, 'D')
    tasks, seen = [], set()
    for row in read_tasks(MANIFEST):
        if row['session_type'] != 'VR' or not row['unit_ids']:
            continue
        mouse = int(str(row['mouse']).upper().removeprefix('M'))
        day = int(str(row['day']).upper().removeprefix('D'))
        if (mice is not None and mouse not in mice) or (days is not None and day not in days):
            continue
        relative = Path(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid relative path: {relative}')
        key = (row['experiment'], mouse, day)
        if key in seen:
            raise ValueError(f'Duplicate session: {key}')
        seen.add(key)
        tasks.append(dict(experiment=row['experiment'], mouse=mouse, day=day,
                          relative_path=str(relative)))
    if not tasks:
        parser.error('No VR sessions match the selection')
    output = args.output_dir.resolve()
    run = output / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    for task in tasks:
        task['nwb_path'] = str(INPUT_ROOT / task['relative_path'])
    manifest = run / 'tasks.jsonl'
    manifest.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    script = Path(__file__).with_name('run_decoder.sh').resolve()
    def submit(phase, options):
        command = ['qsub', '-terse', '-N', f'decode_{phase}',
                   '-o', str(run), '-e', str(run),
                   '-v', f'DECODER_SCRIPT_DIR={script.parent}', *options,
                   str(script), phase, str(manifest), str(output), args.loss_fn,
                   sys.executable]
        reply = subprocess.check_output(command, text=True).strip()
        job_id = reply.split('.')[0]
        if not job_id.isdigit():
            raise RuntimeError(f'Unexpected qsub response: {reply}')
        print(f'{phase}: {reply}', flush=True)
        return job_id
    stage_id = submit('stage', ['-q', 'staging', '-l', 'h_rt=02:00:00'])
    submit('fit', ['-hold_jid', stage_id, '-t', f'1-{len(tasks)}', '-tc', '50'])
    print(f'{len(tasks)} sessions submitted. Results: {output}\nLogs: {run}')


if __name__ == '__main__':
    main()
