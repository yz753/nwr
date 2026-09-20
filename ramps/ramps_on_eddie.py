"""Stage selected NWBs onto scratch before enumerating units and submitting jobs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import os

DEFAULT_SOURCE = '/exports/cmvm/datastore/sbms/groups/INCR-NolanLab/ActiveProjects/Yiming/NWR1/processed'
DEFAULT_OUTPUT = '/exports/eddie/scratch/s2155699/ephys/ramps/ramps_results'
DEFAULT_JSONL = "/exports/eddie/scratch/s2155699/ephys/nwb_units.jsonl"


def parse_ids(value, prefix):
    result = set()
    for item in value.split(','):
        item = item.strip().upper()
        if item.startswith(prefix):
            item = item[1:]
        if not item.isdigit():
            raise ValueError(f'Invalid {prefix} identifier: {item!r}')
        result.add(int(item))
    return result


def discover_nwbs(root, mice=None, days=None, sessions=('VR',)):
    # Match directory identifiers exactly: D2 cannot select D20.
    # Supports both session/file.nwb and session/nwb/file.nwb.
    for mouse in sorted(root.iterdir()):
        match = re.fullmatch(r'M(\d+)', mouse.name)
        if not mouse.is_dir() or not match:
            continue
        if mice is not None and int(match[1]) not in mice:
            continue
        for day in sorted(mouse.iterdir()):
            match = re.fullmatch(r'D(\d+)', day.name)
            if not day.is_dir() or not match:
                continue
            if days is not None and int(match[1]) not in days:
                continue
            for session in sessions:
                folder = day / session
                if folder.is_dir():
                    for nwb in sorted(folder.rglob('*.nwb')):
                        if nwb.is_file() and not nwb.name.startswith('._'):
                            yield nwb.resolve(), mouse.name, day.name, session


def read_unit_ids(path):
    # Standard NWB unit IDs only; dataset-specific data extraction is in the adapter.
    import h5py
    with h5py.File(path, 'r') as f:
        return [] if 'units/id' not in f else [int(x) for x in f['units/id'][:]]


def create_manifest(root, run_dir, mice, days, sessions, read_ids=read_unit_ids):
    count = 0
    files = []
    seen_pairs = set()
    manifest = run_dir / 'tasks.jsonl.partial'
    with manifest.open('w') as out:
        for path, mouse, day, session in discover_nwbs(root, mice, days, sessions):
            ids = read_ids(path)
            if len(ids) != len(set(ids)):
                raise ValueError(f'Duplicate unit IDs in {path}')
            stat = path.stat()
            files.append({'path': str(path), 'mouse': mouse, 'day': day,
                          'session_type': session, 'unit_count': len(ids)})
            print(f'{mouse} {day} {session}: {path.name}: {len(ids)} units', flush=True)
            for unit_id in ids:
                pair = (str(path), unit_id)
                if pair in seen_pairs:
                    raise ValueError(f'Duplicate file/unit: {pair}')
                seen_pairs.add(pair)
                count += 1
                task = {'task_id': count, 'nwb_path': str(path), 'unit_id': unit_id,
                        'mouse': mouse, 'day': day, 'session_type': session,
                        'source_size': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
                        'experiment': root.parent.name}
                out.write(json.dumps(task) + '\n')
    (run_dir / 'files.json').write_text(json.dumps(files, indent=2))
    if not count:
        raise ValueError('No units found for the selection; nothing submitted')
    manifest.replace(run_dir / 'tasks.jsonl')
    return count


def submit(run_dir, count, concurrency, python, hold_jid=None):
    scripts = Path(__file__).resolve().parent
    result = subprocess.check_output([
        'qsub', '-terse', '-t', f'1-{count}', '-tc', str(concurrency),
        *(['-hold_jid', hold_jid] if hold_jid else []),
        '-o', str(run_dir / 'logs'), '-e', str(run_dir / 'logs'),
        str(scripts / 'run_ramps.sh'), str(scripts), str(run_dir), python,
    ], text=True).strip()
    job_id = result.split('.')[0]
    if not job_id.isdigit():
        raise RuntimeError(f'Could not parse qsub response: {result}')
    update_jobs(run_dir, array_job=job_id)
    print(f'Array job {result}; results: {run_dir}')


def save_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def update_jobs(run_dir, **updates):
    # Separate job-ID files avoid concurrent parent/staging writes to one file.
    for key, value in updates.items():
        save_json(run_dir / f'{key}.json', {key: value})


def submit_staging(run_dir, operation, python):
    scripts = Path(__file__).resolve().parent
    result = subprocess.check_output([
        'qsub', '-terse', '-N', f'ramps_{operation}',
        '-o', str(run_dir / 'logs'), '-e', str(run_dir / 'logs'),
        str(scripts / 'stage_ramps.sh'), str(scripts), str(run_dir), python, operation,
    ], text=True).strip()
    job_id = result.split('.')[0]
    if not job_id.isdigit():
        raise RuntimeError(f'Could not parse qsub response: {result}')
    update_jobs(run_dir, **{f'{operation}_job': job_id})
    print(f'{operation} staging job {job_id}; run directory: {run_dir}')
    return job_id


def stage_in(run_dir, read_ids=read_unit_ids):
    config = json.loads((run_dir / 'config.json').read_text())
    if (run_dir / 'stagein_complete.json').exists():
        raise RuntimeError('This run is already staged; do not submit it twice')
    source_root = Path(config['source_root']).resolve(strict=True)
    scratch_root = run_dir / 'input' / config['experiment'] / 'processed'
    selected = json.loads((run_dir / 'selected_files.json').read_text())
    inventory = []
    tasks = []
    # Discovery and copying execute on the staging queue.
    for entry in selected:
        relative = Path(entry['relative_path'])
        source = source_root / relative
        destination = scratch_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        if before.st_size != entry['source_size']:
            raise RuntimeError(f'File size changed; regenerate the unit manifest: {source}')
        subprocess.run([
            'rsync', '-rt', '--no-perms', '--no-owner', '--no-group',
            '--chmod=Du+rwx,Fu+rw', str(source), str(destination),
        ], check=True)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed during staging: {source}')
        if destination.stat().st_size != after.st_size:
            raise RuntimeError(f'Staged file size mismatch: {source}')
        if read_ids(destination) != entry['unit_ids']:
            raise RuntimeError(f'Unit IDs changed; regenerate the unit manifest: {source}')
        stat = destination.stat()
        for unit_id in entry['unit_ids']:
            tasks.append({'task_id': len(tasks) + 1, 'nwb_path': str(destination),
                          'unit_id': unit_id, 'mouse': entry['mouse'], 'day': entry['day'],
                          'session_type': entry['session_type'], 'experiment': config['experiment'],
                          'source_size': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns})
        inventory.append({'source': str(source), 'scratch': str(destination),
                          'mouse': entry['mouse'], 'day': entry['day'], 'session_type': entry['session_type']})
    if not inventory:
        raise RuntimeError('No NWBs match the selection; no array submitted')
    save_json(run_dir / 'staged_files.json', inventory)
    count = len(tasks)
    if count != config['task_count'] or not count:
        raise RuntimeError('Unit count does not match the submitted array')
    temporary = run_dir / 'tasks.jsonl.partial'
    temporary.write_text(''.join(json.dumps(task) + '\n' for task in tasks))
    temporary.replace(run_dir / 'tasks.jsonl')
    save_json(run_dir / 'stagein_complete.json', {'tasks': count, 'nwb_files': len(inventory)})
    print(f'Staging complete: {len(inventory)} NWBs, {count} unit tasks', flush=True)
    # Never call qsub inside a staging job. The launcher already submitted the held array.


def submit_prepared(run_dir):
    config = json.loads((run_dir / 'config.json').read_text())
    marker = json.loads((run_dir / 'stagein_complete.json').read_text())
    if (run_dir / 'array_job.json').exists():
        raise RuntimeError('An array was already submitted for this run')
    submit(run_dir, marker['tasks'], config['max_parallel'], config['python'])


def stage_out(run_dir):
    config = json.loads((run_dir / 'config.json').read_text())
    tasks = [json.loads(line) for line in (run_dir / 'tasks.jsonl').read_text().splitlines()]
    if not tasks:
        raise RuntimeError('Empty task manifest')
    for task in tasks:
        status = json.loads((run_dir / 'tasks' / f"{task['task_id']:08d}" / 'status.json').read_text())
        allowed = status.get('state') == 'complete' or (
            status.get('state') == 'skipped_non_vr' and task.get('session_type') == 'OF')
        if status.get('task') != task or not allowed:
            raise RuntimeError(f"Task {task['task_id']} is unfinished or failed; no stage-out performed")
    if not (run_dir / 'ramps_classification.csv').is_file():
        raise RuntimeError('Run concat_csv.py for this run before staging out')
    destination = Path(config['stageout_root']) / run_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'rsync', '-rt', '--no-perms', '--no-owner', '--no-group',
        '--exclude=/input/', '--exclude=*.tmp',
        str(run_dir) + '/', str(destination) + '/',
    ], check=True)
    print(f'Results copied to {destination}; input NWBs excluded; scratch files retained.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mice')
    p.add_argument('--unit-manifest', type=Path, help='JSONL generated by prepare_nwb_units.py', default=Path(DEFAULT_JSONL))
    p.add_argument('--days')
    p.add_argument('--all', action='store_true')
    p.add_argument('--sessions', default='VR')
    p.add_argument('--data_folder', type=Path, default=Path(DEFAULT_SOURCE))
    p.add_argument('--output_dir', type=Path, default=Path(DEFAULT_OUTPUT))
    p.add_argument('--max_parallel', type=int, default=100)
    p.add_argument('--prepare-only', action='store_true', help='Stage inputs and build manifest without an array')
    p.add_argument('--stageout_dir', type=Path, help='Default: DATASET/ramps_results on DataStore')
    actions = p.add_mutually_exclusive_group()
    actions.add_argument('--stage-out', type=Path, metavar='RUN_DIR', help='Submit a staging job to copy results back')
    actions.add_argument('--submit-run', type=Path, metavar='RUN_DIR', help='Submit an already prepared run')
    actions.add_argument('--stage-in-worker', type=Path, help=argparse.SUPPRESS)
    actions.add_argument('--stage-out-worker', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.stage_in_worker or a.stage_out_worker:
        if not os.environ.get('JOB_ID'):
            p.error('Internal staging workers must be submitted through qsub')
        if a.stage_in_worker:
            stage_in(a.stage_in_worker.resolve())
        else:
            stage_out(a.stage_out_worker.resolve())
        return
    if a.stage_out:
        run_dir = a.stage_out.resolve(strict=True)
        # Read config from scratch only; no DataStore access on the launch node.
        config = json.loads((run_dir / 'config.json').read_text())
        submit_staging(run_dir, 'out', config['python'])
        return
    if a.submit_run:
        submit_prepared(a.submit_run.resolve(strict=True))
        return
    if a.all and (a.mice or a.days):
        p.error('--all cannot be combined with --mice or --days')
    if not a.all and not (a.mice and a.days):
        p.error('Provide --mice and --days, or --all')
    if a.max_parallel < 1:
        p.error('--max_parallel must be positive')
    try:
        mice = None if a.all else sorted(parse_ids(a.mice, 'M'))
        days = None if a.all else sorted(parse_ids(a.days, 'D'))
    except ValueError as error:
        p.error(str(error))
    sessions = list(dict.fromkeys(x.strip() for x in a.sessions.split(',')))
    if any(x not in ('VR', 'OF') for x in sessions):
        p.error('--sessions must contain VR and/or OF')
    if a.unit_manifest is None:
        p.error('--unit-manifest is required for a new run')
    from prepare_nwb_units import read_manifest
    entries = read_manifest(a.unit_manifest)
    selected = [e for e in entries
                if (mice is None or int(e['mouse'][1:]) in mice)
                and (days is None or int(e['day'][1:]) in days)
                and e['session_type'] in sessions and e['unit_ids']]
    if not selected:
        p.error('No units match this selection')
    count = sum(len(e['unit_ids']) for e in selected)
    experiments = {e['experiment'] for e in selected}
    if len(experiments) != 1:
        p.error('Each run must select one experiment')
    # Lexical path handling only: do not resolve/stat/list DataStore here.
    source = Path(os.path.abspath(os.path.expanduser(str(a.data_folder))))
    stageout_root = (Path(os.path.abspath(str(a.stageout_dir))) if a.stageout_dir
                     else source.parent / 'ramps_results')
    run_dir = a.output_dir.resolve() / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True)
    for name in ('logs', 'tasks'):
        (run_dir / name).mkdir()
    save_json(run_dir / 'selected_files.json', selected)
    save_json(run_dir / 'config.json', {
        'source_root': str(source), 'experiment': next(iter(experiments)), 'task_count': count,
        'stageout_root': str(stageout_root), 'mice': mice, 'days': days,
        'sessions': sessions, 'max_parallel': a.max_parallel,
        'prepare_only': a.prepare_only, 'python': sys.executable,
    })
    stage_id = submit_staging(run_dir, 'in', sys.executable)
    if not a.prepare_only:
        submit(run_dir, count, a.max_parallel, sys.executable, hold_jid=stage_id)


if __name__ == '__main__':
    main()
