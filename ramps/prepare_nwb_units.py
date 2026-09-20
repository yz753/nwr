"""Inventory NWB unit IDs where DataStore is accessible; no spike data are loaded."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re


def read_manifest(path):
    entries, seen = [], set()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            entry = json.loads(line)
            relative = PurePosixPath(entry['relative_path'])
            if relative.is_absolute() or '..' in relative.parts or len(relative.parts) < 4:
                raise ValueError(f'Invalid relative NWB path: {relative}')
            mouse, day, session = relative.parts[:3]
            if (not re.fullmatch(r'M\d+', mouse) or not re.fullmatch(r'D\d+', day)
                    or session not in ('VR', 'OF') or relative.suffix != '.nwb'):
                raise ValueError(f'Unexpected NWB layout: {relative}')
            if [entry['mouse'], entry['day'], entry['session_type']] != [mouse, day, session]:
                raise ValueError(f'Metadata does not match path: {relative}')
            experiment = entry['experiment']
            if not experiment or experiment in ('.', '..') or '/' in experiment or '\\' in experiment:
                raise ValueError('Invalid experiment name')
            ids = entry['unit_ids']
            if not isinstance(ids, list) or any(type(x) is not int for x in ids) or len(ids) != len(set(ids)):
                raise ValueError(f'Invalid or duplicate unit IDs: {relative}')
            if type(entry['source_size']) is not int or entry['source_size'] < 0:
                raise ValueError(f'Invalid file size: {relative}')
            if str(relative) in seen:
                raise ValueError(f'Duplicate NWB path: {relative}')
            seen.add(str(relative))
            entries.append(entry)
    if not entries:
        raise ValueError('Empty unit manifest')
    return entries


def prepare(root, output):
    from ramps_on_eddie import discover_nwbs, read_unit_ids
    root = Path(root).resolve(strict=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    files = units = 0
    with temporary.open('w') as stream:
        for path, mouse, day, session in discover_nwbs(root, sessions=('VR', 'OF')):
            before = path.stat()
            ids = read_unit_ids(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'NWB changed during inventory: {path}')
            entry = {'relative_path': path.relative_to(root).as_posix(),
                     'experiment': root.parent.name, 'mouse': mouse, 'day': day,
                     'session_type': session, 'unit_ids': ids, 'source_size': after.st_size}
            stream.write(json.dumps(entry) + '\n')
            files += 1
            units += len(ids)
            print(f'{mouse} {day} {session}: {len(ids)} units', flush=True)
    read_manifest(temporary)
    temporary.replace(output)
    print(f'Saved {files} NWBs / {units} units to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data_folder, args.output)
