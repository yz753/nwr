"""Inventory NWB unit IDs where DataStore is accessible; no spike data are loaded."""

# Allow direct execution with the existing local/Eddie environment.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nwr.io.manifest import (
    Path,
    PurePosixPath,
    argparse,
    discover_nwbs,
    json,
    prepare,
    re,
    read_manifest,
    read_unit_ids,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data_folder, args.output)
