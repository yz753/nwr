'''this aims to replicate the information in fig 1ci in the paper 10.1038/s41467-022-34465-5'''

from pathlib import Path
import argparse
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.stats import zscore



from nwr.io.nwb import get_ramps_input


BLOCK_LABELS = {
    0: "Rewarded",
    1: "Unrewarded",
    2: "Re-rewarded",
}

BLOCK_COLOURS = {
    0: "#d73027",
    1: "#abd9e9",
    2: "#fdae61",
}
    

def corr_cell(
    nwbfile,
    bin_size=0.3,
    smooth_sigma=1,
):
    units = nwbfile["units"]
    unit_ids = np.asarray(list(units.keys()))

    ramps_input = get_ramps_input(
        nwbfile,
        unit_ids[0],
    )

    if ramps_input["training_or_test"] != "test":
        raise ValueError(f"This nwb file is a training day file.")

    position = ramps_input["P"]
    test_blocks = ramps_input["test_block"]

    # compute position tuning curves from rewarded block 0 only, this is used to sort units
    tuning_curves = nap.compute_tuning_curves(
        units,
        features=position,
        epochs=test_blocks[0],
        bins=100,
        range=(0, 200),
    ) # this is for all the units in the nwb file
    tc_values = np.asarray(tuning_curves)

    # sort units based on the position on the track of their peak firing rate
    valid_tc = (
        np.isfinite(tc_values).any(axis=1)
        & (np.nanmax(tc_values, axis=1) > 0)
    )
    peak_bin = np.full(len(unit_ids), np.inf)
    peak_bin[valid_tc] = np.nanargmax(
        tc_values[valid_tc],
        axis=1,
    )
    order = np.argsort(peak_bin)

    # one continuous epoch containing all three blocks
    session_epoch = nap.IntervalSet(
        start=test_blocks[0].start[0],
        end=test_blocks[-1].end[0],
    )

    # time-resolved firing rates
    firing_rate = units.count(
        bin_size,
        ep=session_epoch,
    ) / bin_size
    firing_rate = firing_rate.smooth(
        std=smooth_sigma,
    )

    # Pynapple returns time × unit; the heatmap needs unit × time
    activity = np.asarray(firing_rate.values).T
    # z score it
    activity = zscore(
        activity,
        axis=1,
        nan_policy="omit",
    )

    return (
        activity[order],
        unit_ids[order],
        np.asarray(firing_rate.t),
        test_blocks,
    )


def plot_heatmap(
    activity,
    times,
    test_blocks,
    reward_data,
    output,
    title,
):
    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.6, 0.45, 10],
        width_ratios=[1, 0.025],
        hspace=0.03,
        wspace=0.02,
    )
    reward_ax = fig.add_subplot(grid[0, 0])
    block_ax = fig.add_subplot(
        grid[1, 0],
        sharex=reward_ax,
    )
    ax = fig.add_subplot(
        grid[2, 0],
        sharex=reward_ax,
    )
    cbar_ax = fig.add_subplot(grid[2, 1])

    session_start = test_blocks[0].start[0]
    relative_times = times - session_start
    
    session_end = test_blocks[2].end[-1]

    reward_colours = {
        "reward_received": "black",
        "reward_failed": "red",
    }

    for key, (reward_times, reward_values) in reward_data.items():
        within_session = (
            (reward_times >= session_start)
            & (reward_times <= session_end)
        )

        reward_ax.plot(
            reward_times[within_session] - session_start,
            reward_values[within_session],
            label=key,
            color=reward_colours[key],
            linewidth=0.35,
            drawstyle="steps-post",
        )

    reward_ax.set_xlim(
        relative_times[0],
        relative_times[-1],
    )

    reward_ax.set_ylim(-0.05, 1.05)

    # Remove all four borders.
    for spine in reward_ax.spines.values():
        spine.set_visible(False)

    # Remove x-axis and y-axis ticks, values and labels.
    reward_ax.set_xticks([])
    reward_ax.set_yticks([])
    reward_ax.set_xlabel("")
    reward_ax.set_ylabel("")

    # Remove tick marks completely.
    reward_ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    # Remove the legend if one was previously created.
    legend = reward_ax.get_legend()
    if legend is not None:
        legend.remove()

    for block in (0, 1, 2):
        start = test_blocks[block].start[0]
        end = test_blocks[block].end[0]
        start -= session_start
        end -= session_start

        block_ax.axvspan(
            start,
            end,
            color=BLOCK_COLOURS[block],
        )

        block_ax.text(
            (start + end) / 2,
            0.5,
            BLOCK_LABELS[block],
            ha="center",
            va="center",
            color="black",
        )

    block_ax.set_xlim(
        relative_times[0],
        relative_times[-1],
    )
    block_ax.set_ylim(0, 1)
    block_ax.axis("off")

    limit = np.nanpercentile(
        np.abs(activity),
        99,
    )

    if not np.isfinite(limit) or limit == 0:
        limit = 1.0

    image = ax.imshow(
        activity,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=-limit,
        vmax=limit,
        extent=(
            relative_times[0],
            relative_times[-1],
            len(activity),
            0,
        ),
    )

    ax.set(
        xlabel="time (s)",
        ylabel="sorted units by tc in block 0",
    )

    fig.colorbar(
        image,
        cax=cbar_ax,
        label="z-scored firing rate",
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)




