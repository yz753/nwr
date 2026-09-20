#!/usr/bin/env python3
"""Distributed GLM-HMM model selection for the J6 D43 VR NWB dataset.

The program has three stages:

1. ``prepare`` streams the NWB file once and saves aligned NumPy arrays.
2. ``fit-task`` evaluates one basis/state configuration on one CV fold.
3. ``aggregate`` combines fold scores, tests each model's winner, and saves plots.

The companion SGE script submits these stages with scheduler dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODEL_NAMES = (
    "position_only",
    "trial_type_only",
    "speed_only",
    "position_trial_type",
    "position_speed",
    "position_speed_trial_type",
)
SPEED_MODELS = {"speed_only", "position_speed", "position_speed_trial_type"}

DEFAULT_BIN_SIZE = 0.02

DEFAULT_UNIT_JSONL_PATH = "/exports/eddie/scratch/s2155699/ephys/nwr/nwb_units.jsonl"


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    model_name: str
    n_position_basis: int
    n_speed_basis: int | None
    n_states: int


def candidate_table(args: argparse.Namespace) -> list[Candidate]:
    """Return every complete model/basis/state candidate in stable order."""
    candidates: list[Candidate] = []
    candidate_id = 0
    position_sizes = range(args.position_basis_min, args.position_basis_max)
    speed_sizes = range(args.speed_basis_min, args.speed_basis_max)
    state_sizes = range(args.states_min, args.states_max)

    for model_name in MODEL_NAMES:
        for n_position in position_sizes:
            model_speed_sizes: list[int | None]
            if model_name in SPEED_MODELS:
                model_speed_sizes = list(speed_sizes)
            else:
                model_speed_sizes = [None]

            for n_speed in model_speed_sizes:
                for n_states in state_sizes:
                    candidates.append(
                        Candidate(
                            candidate_id=candidate_id,
                            model_name=model_name,
                            n_position_basis=n_position,
                            n_speed_basis=n_speed,
                            n_states=n_states,
                        )
                    )
                    candidate_id += 1
    print('Prepared all candidate models to be fitted.', flush=True)
    return candidates


def task_count(args: argparse.Namespace) -> int:
    return len(candidate_table(args)) * args.cv_folds


def task_to_candidate(
    args: argparse.Namespace, task_id: int
) -> tuple[Candidate, int]:
    """Map a one-based SGE task ID to a candidate and zero-based CV fold."""
    zero_based = task_id - 1
    total = task_count(args)
    if zero_based < 0 or zero_based >= total:
        raise ValueError(f"task_id must be in [1, {total}], got {task_id}")
    candidate_index, fold_index = divmod(zero_based, args.cv_folds)
    return candidate_table(args)[candidate_index], fold_index


def build_basis(
    candidate: Candidate,
    position_bounds: tuple[float, float],
    speed_bounds: tuple[float, float],
    ):
    """Build a four-input composite basis for one candidate."""
    position = (
        nmo_basis.BSplineEval(
            n_basis_funcs=candidate.n_position_basis,
            bounds=position_bounds,
            label="position",
        )
        if candidate.n_position_basis is not None
        else nmo_basis.Zero()
    )
    speed = (
        nmo_basis.BSplineEval(
            n_basis_funcs=int(candidate.n_speed_basis),
            bounds=speed_bounds,
            label="speed",
        )
        if candidate.n_speed_basis is not None
        else nmo_basis.Zero()
    )
    trial_type = (
        nmo_basis.IdentityEval(label="trial_type")
        if candidate.model_name
        in {"trial_type_only", "position_trial_type", "position_speed_trial_type"}
        else nmo_basis.Zero()
    )

    composite = position + speed + trial_type
    composite.label = candidate.model_name
    composite.set_input_shape(1, 1, 1)
    return composite


def make_model(candidate: Candidate, args: argparse.Namespace) -> GLMHMM:
    return GLMHMM(
        n_states=candidate.n_states,
        observation_model='Poisson',
        regularizer="Ridge",
        seed=jax.random.PRNGKey(args.seed),
        solver_name="LBFGS",
        tol=args.tol,
        maxiter=args.maxiter,
    )


def prepare(args: argparse.Namespace) -> None:
    """Stream the NWB once and save the arrays required by all fit tasks."""
    output = Path(args.data_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(args.nwb_file, "r", load_namespaces=True) as io:
        nwb = nap.NWBFile(io.read())
        position = nwb["P"]
        speed = nwb["S"]
        trials = nwb["trials"]
        trial_number_tsd = nwb["trial_number"]
        trial_type_tsd = nwb["trial_type"]
        neural_counts = nwb["units"][args.unit_index].count(bin_size=args.bin_size)
        
        # fs of behaviour & ephys count should be the same
        down_position = position.interpolate(
            neural_counts, ep=neural_counts.time_support
        )
        down_speed = speed.interpolate(neural_counts, ep=neural_counts.time_support)
        down_trial_type = trial_type_tsd.interpolate(
            neural_counts, ep=neural_counts.time_support
        )
        down_trial_number = trial_number_tsd.interpolate(
            neural_counts, ep=neural_counts.time_support
        )
        
        times = np.asarray(neural_counts.t, dtype=float)
        
        X = np.column_stack(
            [
                np.asarray(down_position),
                np.asarray(down_speed),
                np.asarray(down_trial_type),
            ]
        )
        y = np.asarray(neural_counts, dtype=float)
        
        unique_types = np.array(["b", "nb"])
        trial_type_by_trial = np.asarray(
            [
                {value: index for index, value in enumerate(unique_types)}[value]
                for value in np.asarray(trials["type"])
            ],
            dtype=int,
        )

        np.savez_compressed(
            output,
            X=X,
            y=y,
            times=times,
            sampling_rate=np.asarray(neural_counts.rate),
            time_support=np.asarray(neural_counts.time_support),
            unit_index=np.asarray(args.unit_index),
            position_times=np.asarray(position.t),
            position_values=np.asarray(position),
            speed_times=np.asarray(speed.t),
            speed_values=np.asarray(speed),
            spike_times=np.asarray(nwb["units"][args.unit_index].t),
            trial_starts=np.asarray(trials["start"]),
            trial_ends=np.asarray(trials["end"]),
            moving_starts=np.asarray(nwb["moving"]['start']),
            moving_ends=np.asarray(nwb["moving"]['end']),
            trial_type_by_trial=trial_type_by_trial,
            downsampled_trial_number=np.asarray(down_trial_number),
        )
        print(f"Processed nwb file and saved to {output}.", flush=True)


def fit_task(args: argparse.Namespace) -> None:
    """Fit and score one candidate on one fold."""
    candidate, fold_index = task_to_candidate(args, args.task_id)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"task_{args.task_id:05d}.json"

    with np.load(args.data_file) as data:
        X = data["X"]
        y = data["y"]

    split_index = int((1.0 - args.test_fraction) * len(X))
    X_grid = X[:split_index]
    y_grid = y[:split_index]
    position_bounds = (
        float(np.nanmin(X_grid[:, 0])), 
        float(np.nanmax(X_grid[:, 0]))
        )
    speed_bounds = (
        float(np.nanmin(X_grid[:, 1])), 
        float(np.nanmax(X_grid[:, 1]))
        )
    
    train_index, validation_index = list(
        KFold(n_splits=args.cv_folds, shuffle=False).split(X_grid)
    )[fold_index]

    started = time.time()
    transformer = build_basis(
        candidate, 
        position_bounds=position_bounds, 
        speed_bounds=speed_bounds
        ).to_transformer()
    X_train = transformer.fit_transform(X_grid[train_index])
    X_validation = transformer.transform(X_grid[validation_index])
    model = make_model(candidate, args)
    model.fit(X_train, y_grid[train_index])
    score = float(model.score(X_validation, y_grid[validation_index]))
    
    # calculate AIC score
    K = candidate.n_states
    p = X_validation.shape[1]
    AIC_score = -2 * np.log(score) + 2 * (K * (p+1) + K * (K-1) + (K-1))

    result: dict[str, Any] = {
        **asdict(candidate),
        "task_id": args.task_id,
        "fold": fold_index,
        "score": score,
        "AIC_score": AIC_score,
        "n_train": int(len(train_index)),
        "n_validation": int(len(validation_index)),
        "elapsed_seconds": time.time() - started,
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(result_path)
    print(json.dumps(result))
    print('Completed task:', args.task_id, ', candidate:', candidate.model_name, ', fold:', fold_index, ', score:', score, flush=True)


def aggregate(args: argparse.Namespace) -> None:
    """Select and test the best candidate separately for every model."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = task_count(args)
    paths = [Path(args.results_dir) / f"task_{i:05d}.json" for i in range(1, expected + 1)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} task results; first: {missing[:5]}")

    fold_df = pd.DataFrame([json.loads(path.read_text()) for path in paths])
    fold_df.to_csv(output_dir / "fold_results.csv", index=False)

    group_columns = [
        "candidate_id",
        "model_name",
        "n_position_basis",
        "n_speed_basis",
        "n_states",
    ]
    summary = (
        fold_df.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            mean_test_score=("score", "mean"),
            std_test_score=("score", "std"),
            total_fit_seconds=("elapsed_seconds", "sum"),
        )
        .sort_values("mean_test_score", ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(0, "rank_test_score", np.arange(1, len(summary) + 1))
    summary["rank_within_model"] = (
        summary.groupby("model_name")["mean_test_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    summary.to_csv(output_dir / "all_cv_results.csv", index=False)

    with np.load(args.data_file) as data:
        X = data["X"]
        y = data["y"]
        times = data["times"]
        sampling_rate = float(data["sampling_rate"])
        position = nap.Tsd(t=data["position_times"], d=data["position_values"])
        actual_spikes = nap.TsGroup(
            {int(data["unit_index"]): nap.Ts(t=data["spike_times"])}
        )
        trial_starts = data["trial_starts"]
        trial_ends = data["trial_ends"]
        trial_type_by_trial = data["trial_type_by_trial"]
        trial_number = data["downsampled_trial_number"]
        moving_starts = data["moving_starts"]
        moving_ends = data["moving_ends"]
        unit_index = int(data["unit_index"])
        
    split_index = int((1.0 - args.test_fraction) * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    candidates = candidate_table(args)
    best_rows = summary.loc[
        summary.groupby("model_name")["mean_test_score"].idxmax()
    ].sort_values("model_name")

    model_dir = output_dir / "best_models"
    model_output_dir = output_dir / "best_model_outputs"
    plot_dir = output_dir / "plots"
    model_dir.mkdir(exist_ok=True)
    model_output_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    trial_type_intervalset = nap.IntervalSet(
        pd.DataFrame({"start": trial_starts, "end": trial_ends})
    )
    plot_bin_size = 300

    def compute_tuning_curve(firing_data, moving_starts, moving_ends, unit_index):
        if isinstance(firing_data, nap.Ts):
            firing_data = nap.TsGroup({
                unit_index: firing_data[unit_index]
            })
        elif isinstance(firing_data, nap.Tsd):
            pass
        
        # construct moving intervalset
        moving_intervalset = nap.IntervalSet(
            pd.DataFrame({"start": moving_starts, "end": moving_ends})
        )
        
        return nap.compute_tuning_curves(
            firing_data,
            position,
            bins=plot_bin_size,
            range=(0, 300),
            epochs=firing_data.time_support.intersect(moving_intervalset)
            .intersect(trials_for_plot),
        )[0]

    best_results: dict[str, dict[str, Any]] = {}
    best_records: list[dict[str, Any]] = []
    for _, best_row in best_rows.iterrows():
        candidate = candidates[int(best_row["candidate_id"])]

        # Fit only the outer training data, then score the untouched outer test data.
        position_bounds = (
            float(np.nanmin(X_train[:, 0])), 
            float(np.nanmax(X_train[:, 0]))
            )
        speed_bounds = (
            float(np.nanmin(X_train[:, 1])), 
            float(np.nanmax(X_train[:, 1]))
            )
        train_transformer = build_basis(
            candidate, 
            position_bounds=position_bounds, 
            speed_bounds=speed_bounds
            ).to_transformer()
        X_train_design = train_transformer.fit_transform(X_train)
        X_test_design = train_transformer.transform(X_test)
        train_model = make_model(candidate, args)
        train_model.fit(X_train_design, y_train)
        held_out_test_score = float(train_model.score(X_test_design, y_test))

        # Refit the selected parameters on all data solely for full-session prediction.
        position_bounds = (
            float(np.nanmin(X[:, 0])), 
            float(np.nanmax(X[:, 0]))
            )
        speed_bounds = (
            float(np.nanmin(X[:, 1])), 
            float(np.nanmax(X[:, 1]))
            )
        full_transformer = build_basis(
            candidate, 
            position_bounds=position_bounds, 
            speed_bounds=speed_bounds
            ).to_transformer()
        X_design = full_transformer.fit_transform(X)
        model = make_model(candidate, args)
        model.fit(X_design, y)
        _, predicted_rate, simulated_states = model.simulate(
            jax.random.PRNGKey(args.seed),
            X_design,
            state_format="one-hot",
        )
        downsampled_predicted_rate = predicted_rate * sampling_rate
        posteriors = model.smooth_proba(X_design, y)
        most_probable_states = nap.Tsd(
            t=posteriors.t,
            d=posteriors.values.argmax(axis=1),
            time_support=posteriors.time_support
        )
        
        # calculate AIC score
        K = candidate.n_states
        p = X_design.shape[1]
        held_out_AIC_score = -2 * np.log(held_out_test_score) + 2 * (K * (p+1) + K * (K-1) + (K-1))

        model.save_params(model_dir / f"{candidate.model_name}.npz")
        np.savez_compressed(
            model_output_dir / f"{candidate.model_name}.npz",
            times=times,
            downsampled_predicted_rate=downsampled_predicted_rate,
            simulated_states=np.asarray(simulated_states),
            posterior_probabilities=np.asarray(posteriors),
            most_probable_states=np.asarray(most_probable_states),
            trial_number_tsd=trial_number
        )

        record = {
            **asdict(candidate),
            "mean_cv_score": float(best_row["mean_test_score"]),
            "std_cv_score": float(best_row["std_test_score"]),
            "held_out_test_score": held_out_test_score,
            'held_out_AIC_score': held_out_AIC_score,
            "n_outer_train": int(len(y_train)),
            "n_outer_test": int(len(y_test)),
            "n_design_features": int(X_design.shape[1]),
        }
        best_records.append(record)
        best_results[candidate.model_name] = record

        predicted_rate_tsd = nap.Tsd(
            t=times, 
            d=downsampled_predicted_rate,
            time_support=nap.IntervalSet(
                start=np.asarray(data["time_support"].time_support)[0][0], 
                end=np.asarray(data["time_support"].time_support)[0][1]
                )
            )
        model_plot_dir = plot_dir / candidate.model_name
        model_plot_dir.mkdir(exist_ok=True)
        for trial_type_value in [[0], [1], [2]]:
            trials_for_plot = trial_type_intervalset[
                trial_type_by_trial.isin(trial_type_value)
            ]
            predicted_tc = compute_tuning_curve(predicted_rate_tsd, moving_starts, moving_ends, unit_index)
            actual_tc = compute_tuning_curve(actual_spikes, moving_starts, moving_ends, unit_index)
            plt.figure(figsize=(10, 5))
            plt.plot(
                predicted_tc.coords["0"].values,
                predicted_tc.values,
                label="GLM-HMM",
                color="blue",
            )
            plt.plot(
                actual_tc.coords["0"].values,
                actual_tc.values,
                label="recorded",
                color="orange",
            )
            plt.title(f"tc, trial type {trial_type_value[0]}")
            plt.xlabel("Position (cm)")
            plt.ylabel("Firing Rate (Hz)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(model_plot_dir / f"trial_{trial_type_value[0]}.png")
            plt.close()

    pd.DataFrame(best_records).to_csv(output_dir / "best_by_model.csv", index=False)
    (output_dir / "best_by_model.json").write_text(
        json.dumps(best_results, indent=2) + "\n"
    )
    print(json.dumps(best_results, indent=2))
    print('Found optimal params for each model and finished refitting and plotting.', flush=True)


def add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--position-basis-min", type=int, default=6)
    parser.add_argument("--position-basis-max", type=int, default=13)
    parser.add_argument("--speed-basis-min", type=int, default=4)
    parser.add_argument("--speed-basis-max", type=int, default=11)
    parser.add_argument("--states-min", type=int, default=2)
    parser.add_argument("--states-max", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--maxiter", type=int, default=800)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--nwb-file", required=True)
    prepare_parser.add_argument("--unit-index", type=int, default=0)
    prepare_parser.add_argument("--bin-size", type=float, default=DEFAULT_BIN_SIZE)
    prepare_parser.add_argument("--data-file", required=True)

    fit_parser = subparsers.add_parser("fit-task")
    add_grid_arguments(fit_parser)
    fit_parser.add_argument("--task-id", type=int, required=True)
    fit_parser.add_argument("--data-file", required=True)
    fit_parser.add_argument("--results-dir", required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    add_grid_arguments(aggregate_parser)
    aggregate_parser.add_argument("--data-file", required=True)
    aggregate_parser.add_argument("--results-dir", required=True)
    aggregate_parser.add_argument("--output-dir", required=True)

    count_parser = subparsers.add_parser("task-count")
    add_grid_arguments(count_parser)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "task-count":
        print(task_count(args))
        return
    
    global jax, jnp, plt, np, pd, nap, nmo_basis, GLMHMM, NWBHDF5IO, KFold
    
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pynapple as nap
    from nemos import basis as nmo_basis
    from nemos.glm_hmm import GLMHMM
    from pynwb import NWBHDF5IO
    from sklearn.model_selection import KFold
    
    if args.command == "prepare":
        prepare(args)
    elif args.command == "fit-task":
        fit_task(args)
    elif args.command == "aggregate":
        aggregate(args)
    elif args.command == "task-count":
        print(task_count(args))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
