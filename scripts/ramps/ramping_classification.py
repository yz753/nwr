# Allow direct execution with the existing local/Eddie environment.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nwr.ramps.classification import classify
"""Run all trial/track combinations for one NWB unit selected by manifest row."""
import argparse
import json
from pathlib import Path
import time
import traceback
import pynapple as nap


def atomic_json(path, obj):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def get_task(run_dir, task_id):
    if task_id < 1:
        raise ValueError('task_id must be positive')
    with (run_dir / 'tasks.jsonl').open() as f:
        for index, line in enumerate(f, 1):
            if index == task_id:
                task = json.loads(line)
                if task['task_id'] != task_id:
                    raise ValueError('Manifest row/task ID mismatch')
                return task
    raise ValueError(f'No manifest row {task_id}')




def run_task(run_dir, task_id, classifier=classify):
    task = get_task(run_dir, task_id)
    out = run_dir / 'tasks' / f'{task_id:08d}'
    out.mkdir(parents=True, exist_ok=True)
    status_file = out / 'status.json'
    start = time.monotonic()
    status = {'task': task, 'state': 'running'}
    atomic_json(status_file, status)
    try:
        rows, errors, expected, state = classifier(task, out)
        status.update(state=state, rows=len(rows), expected_rows=expected, errors=errors)
    except Exception:
        status.update(state='failed', rows=0, errors=[{'traceback': traceback.format_exc()}])
        print(status['errors'][0]['traceback'], flush=True)
    status['elapsed_seconds'] = time.monotonic() - start
    atomic_json(status_file, status)
    return 0 if status['state'] in ('complete', 'skipped_non_vr') else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', type=Path, required=True)
    p.add_argument('--task_id', type=int, required=True)
    a = p.parse_args()
    return run_task(a.run_dir.resolve(), a.task_id)


if __name__ == '__main__':
    raise SystemExit(main())
