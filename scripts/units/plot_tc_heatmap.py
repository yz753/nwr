'''this aims to replicate the information in fig 1ci in the paper 10.1038/s41467-022-34465-5'''

# Allow direct execution with the existing local/Eddie environment.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nwr.units.heatmaps import (
    BLOCK_COLOURS,
    BLOCK_LABELS,
    NWBHDF5IO,
    Path,
    argparse,
    corr_cell,
    get_ramps_input,
    matplotlib,
    nap,
    np,
    plot_heatmap,
    plt,
    sys,
    zscore,
)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nwb-file",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    nwb_path = Path(args.nwb_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = nap.NWBFile(io.read())

        activity, _, times, test_blocks = corr_cell(
            nwbfile,
            bin_size=args.bin_size,
            smooth_sigma=args.smooth_sigma,
        )
        
        reward_data = {}
        for key in ("reward_received", "reward_failed"):
            tsd = nwbfile[key]

            values = np.asarray(
                tsd.values[:],
                dtype=float,
            ).squeeze()

            values[values == 2] = 1

            reward_data[key] = (
                np.asarray(tsd.t, dtype=float),
                values,
            )

    output = (
        output_dir
        / f"{nwb_path.stem}_tc_heatmap.png"
    )

    plot_heatmap(
        activity,
        times,
        test_blocks,
        reward_data,
        output,
        nwb_path.stem,
    )

    print(f"Saved: {output}", flush=True)

if __name__ == "__main__":
    main()
