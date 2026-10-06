#!/usr/bin/env python3

'''this script looks into all job folders (run_date*) and finds the latest complete session for each experiment/mouse/day/session_type combination.
It guides the stageout bash script to select the latest complete results for each session.
For example, if M01 D01 is run both on 2026-10-06 and 2026-10-07, and both are complete, only the 2026-10-07 results will be staged out. If the 2026-10-07 run is incomplete, the 2026-10-06 results will be staged out instead.'''


import json
import sys
from pathlib import Path


results_root = Path(sys.argv[1])
files_to_stageout_list = Path(sys.argv[2])

latest = {}

# The timestamp format in run_YYYYMMDD_HHMMSS_* sorts chronologically.
for run_dir in sorted(results_root.glob("run_*")):
    tasks_file = run_dir / "tasks.jsonl"
    if not tasks_file.exists():
        continue

    sessions = {}

    with tasks_file.open() as f:
        for line in f:
            if not line.strip():
                continue

            task = json.loads(line)
            key = (
                task["experiment"],
                task["mouse"],
                task["day"],
                task["session_type"],
            )
            sessions.setdefault(key, []).append(task)

    for key, tasks in sessions.items():
        complete = all(
            any(
                (
                    run_dir
                    / "tasks"
                    / f"{int(task['task_id']):08d}"
                ).glob("*_results.csv")
            )
            for task in tasks
        )

        if complete:
            latest[key] = (run_dir, tasks)
        else:
            print(
                f"Skipping incomplete session {key} in {run_dir.name}",
                file=sys.stderr,
            )

with files_to_stageout_list.open("w") as f:
    for key, (run_dir, tasks) in sorted(latest.items()):
        experiment, mouse, day, session_type = key

        print(
            f"Selected {run_dir.name}: "
            f"{experiment} {mouse} {day} {session_type}"
        )

        for task in tasks:
            task_dir = (
                run_dir
                / "tasks"
                / f"{int(task['task_id']):08d}"
            )

            print(
                task_dir,
                mouse,
                day,
                session_type,
                run_dir,
                sep="\t",
                file=f,
            )

print(f"Generated {files_to_stageout_list}")