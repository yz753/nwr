'''this is similar to glm_hmm, except the test_block is an input rather than inferred from decoded neural states'''

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from itertools import combinations


single_predictors = [
    "position",
    "trial_type",
    "speed",
    "test_block",
    ]

MODEL_NAMES = []
for single_predictor in range(1, len(single_predictors)+1):
    for predictor_combination in combinations(single_predictors, single_predictor):
        MODEL_NAMES.append('_'.join(predictor_combination))

POSITION_MODELS = {item for item in MODEL_NAMES if "position" in item}
SPEED_MODELS = {item for item in MODEL_NAMES if "speed" in item}
TRIAL_TYPE_MODELS = {item for item in MODEL_NAMES if "trial_type" in item}
TEST_BLOCK_MODELS = {item for item in MODEL_NAMES if "test_block" in item}


DEFAULT_BIN_SIZE = 0.02

DEFAULT_UNIT_JSONL_PATH = "/exports/eddie/scratch/s2155699/ephys/nwr/nwb_units.jsonl"


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    model_name: str
    n_position_basis: int | None
    n_speed_basis: int | None
    n_states: int


def candidate_table(args: argparse.Namespace) -> list[Candidate]:
    """Return every complete model/basis/state candidate in stable order."""
    candidates: list[Candidate] = []
    position_sizes = range(args.position_basis_min, args.position_basis_max)
    speed_sizes = range(args.speed_basis_min, args.speed_basis_max)
    state_sizes = range(args.states_min, args.states_max)

    for model_name in MODEL_NAMES:
        model_position_sizes = (
            position_sizes if model_name in POSITION_MODELS else [None]
        )
        model_speed_sizes = (
            speed_sizes if model_name in SPEED_MODELS else [None]
        )
        for n_position in model_position_sizes:
            for n_speed in model_speed_sizes:
                for n_states in state_sizes:
                    candidates.append(
                        Candidate(
                            candidate_id=len(candidates),
                            model_name=model_name,
                            n_position_basis=n_position,
                            n_speed_basis=n_speed,
                            n_states=n_states,
                        )
                    )
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
        in TRIAL_TYPE_MODELS
        else nmo_basis.Zero()
    )
    test_block = (
        nmo.basis.IdentityEval(label="test_block")
        if candidate.model_name
        in TEST_BLOCK_MODELS
        else nmo_basis.Zero()
    )

    composite = position + speed + trial_type + test_block
    composite.label = candidate.model_name
    composite.set_input_shape(1, 1, 1, 1)
    return composite


def make_model(candidate: Candidate, args: argparse.Namespace) -> PopulationGLM:
    return PopulationGLM(
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
        test_block_tsd = nwb['block_idx']
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
        down_test_block = test_block_tsd.interpolate(
            neural_counts, ep=neural_counts.time_support
        )
        
        times = np.asarray(neural_counts.t, dtype=float)
        
        X = np.column_stack(
            [
                np.asarray(down_position),
                np.asarray(down_speed),
                np.asarray(down_trial_type),
                np.asarray(down_test_block),
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
    AIC_score = -2 * score + 2 * (K * (p+1) + K * (K-1) + (K-1))

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
