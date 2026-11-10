"""Decode position from all neural activity for each session.

Multiple decoders can be used: bayesian from pynapple, or a deep neural network with a single hidden layer and ReLU activation function.
2 forms of tests: 1. train a decoder on 80% of all test blocks and test on the remaining 20%, this aims to answer: is one single decoder able to map all 3 reward conditions?
2. train a decoder on 80% block 0 data only, and test on the remaining 20% block 0 data, all block 1, and all block 2 data, this aims to answer: is a decoder trained on block 0 able to generalise to block 1 and block 2?

"""

import pynapple as nap
import numpy as np
from pynwb import NWBHDF5IO
from pathlib import Path
import traceback, sys
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ramps.nwb_ramps_input import get_ramps_input


def prepare_trials(nwb, behaviour, bin_size):
    """bin spikes within individual trials and return a big list of trial dicts"""
    clusters = nwb["units"]
    position = behaviour["P"]

    # downsample position
    metadata = pd.DataFrame({
        "time": position.t,
        "trial": position.value_from(behaviour["trial_idx"]).d,
        "trial_type": position.value_from(nwb["trial_type"]).d,
        "block": (
            position.value_from(nwb["block_idx"]).d
            if "block_idx" in nwb.keys()
            else np.zeros(len(position))
        ),
    }).dropna()
    
    for column in ("trial", "trial_type", "block"):
        metadata[column] = metadata[column].astype(int)
    
    trials = []

    for trial_id, frame in metadata.groupby("trial", sort=False):
        start, end = frame["time"].iloc[[0, -1]]
        # ditch trials that are extremely short
        if end - start < bin_size:
            continue
        
        # define moving interval epochs for each trial
        epochs = (
            nap.IntervalSet(start=start, end=end)
            .intersect(clusters.time_support)
            .intersect(position.time_support)
            .intersect(behaviour["moving"])
            .drop_short_intervals(bin_size)
        )
        if len(epochs) == 0:
            continue

        counts = clusters.count(bin_size, ep=epochs)
        
        # get ground truth position to be compared with decoded position
        true_position = counts.value_from(position).d

        trials.append({
            "trial": int(trial_id),
            "trial_type": int(frame["trial_type"].iloc[0]),
            "block": int(frame["block"].iloc[0]),
            "epochs": epochs,
            "time": counts.t,
            "X": counts.d.astype(np.float32),
            "position": np.asarray(true_position),
        })

    return trials


class InsufficientTrials(ValueError):
    '''it's possible that we have data shortage for beaconed trials, in this case we report a skip'''



def split_trials(trials, seed, scheme='mixed', test_fraction=0.2):
    """split training & test trials
    80% of trials for each block are randomly selected for training"""
    if scheme not in {"mixed", "block0_only"}:
        raise ValueError(f"Unknown scheme: {scheme}")
    if not trials:
        raise InsufficientTrials("not enough trials: no eligible trials")
    
    blocks = sorted({t["block"] for t in trials}) if scheme == "mixed" else [0]
    train, test = [], []
    for block in blocks:
        candidates = sorted((t for t in trials if t["block"] == block),
                            key=lambda t: t["time"][0])
        if len(candidates) < 2:
            raise InsufficientTrials(f"not enough trials for block {block}: "
                                        f"{len(candidates)} available; need at least 2")
        
        # random shuffle
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(block)]))
        order = rng.permutation(len(candidates))
        n_test = min(len(candidates) - 1,
                        max(1, int(np.ceil(test_fraction * len(candidates)))))
        test.extend(candidates[i] for i in order[:n_test])
        train.extend(candidates[i] for i in order[n_test:])
        
    if scheme == "block0_only":
        test.extend(t for t in trials if t["block"] != 0)
        
    return train, sorted(test, key=lambda t: t["time"][0])


def combine_epochs(trials):
    return nap.IntervalSet(
        start=np.concatenate([trial["epochs"].start for trial in trials]),
        end=np.concatenate([trial["epochs"].end for trial in trials]),
    )
    
    
def position_wasserstein_loss(logits, target_bins, bin_width, reduction='mean'):
    '''cross entropy loss doesn't penalise higher distance between predicted and true position, this loss function does'''
    probability = torch.softmax(logits, dim=1)

    target = F.one_hot(
        target_bins,
        num_classes=logits.shape[1],
    ).to(probability.dtype)

    predicted_cdf = probability.cumsum(dim=1)
    target_cdf = target.cumsum(dim=1)

    loss = (
        torch.abs(predicted_cdf - target_cdf).sum(dim=1)
        * bin_width
    )
    
    return loss.mean() if reduction == 'mean' else loss
    

def fit_deepnet(train_trials, 
                edges, 
                loss_fn: str,
                seed, 
                n_epochs=200):
    """fit a categorical position decoder using training trials only."""
    if loss_fn not in {"cross_entropy", "wasserstein"}:
        raise ValueError(f"Unknown loss function: {loss_fn}. Select from 'cross_entropy' or 'wasserstein'.")
    
    torch.manual_seed(seed)

    X = np.concatenate([trial["X"] for trial in train_trials])
    position = np.concatenate([trial["position"] for trial in train_trials])

    # the number of trials is usually: block 0 > block 2 > block 1
    # which means unrewarded trials are underrepresented
    # the difference is not big but I used weighted sampling because testing fitness on block 1 is important for my study
    block = np.concatenate(
        [
            np.full(len(trial['X']), trial['block'])
            for trial in train_trials
        ]
    )
    _, block_index, block_counts = np.unique(
        block, return_inverse=True, return_counts=True
    )
    weights = len(block) / (
        len(block_counts) * block_counts[block_index]
    ) # length = number of samples (len(X))
    
    labels = np.searchsorted(edges, position, side="right") - 1
    labels = np.clip(labels, 0, len(edges) - 2)

    # z score spike counts to remove the big effect of active neurons on gradient descent
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0 # prevent from dividing by 0

    dataset = TensorDataset(
        torch.as_tensor((X - mean) / std, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
        torch.as_tensor(weights, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = nn.Sequential(
        nn.Linear(X.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, len(edges) - 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction='none')

    model.train()
    
    # right now there's no early stopping
    for _ in range(n_epochs):
        for X_batch, y_batch, w_batch in loader:
            optimizer.zero_grad()
            scores = model(X_batch)
            
            if loss_fn == "cross_entropy":
                loss = criterion(scores, y_batch)
            else:
                loss = position_wasserstein_loss(
                    scores, 
                    y_batch, 
                    bin_width=edges[1] - edges[0],
                    reduction='none',
                )
            weighted_loss = (loss * w_batch).mean()
            weighted_loss.backward()
            optimizer.step()

    model.eval()
    return model, mean, std


def predict_deepnet(fitted, X, centres):
    """return the most probable position and probabilities over bins.
    centres: because positions are in bins, can only take the centre of each position bin to give the best resolution"""
    model, mean, std = fitted
    inputs = torch.as_tensor((X - mean) / std, dtype=torch.float32)

    # Batch inference to limit memory use.
    with torch.no_grad():
        probability = torch.cat([
            torch.softmax(model(batch), dim=1)
            for batch in inputs.split(4096)
        ]).numpy()

    decoded = centres[probability.argmax(axis=1)]
    return decoded, probability


def summarise_trials(samples, metadata):
    summary = (samples.assign(squared_error=samples.signed_error_cm ** 2)
               .groupby("trial", as_index=False)
               .agg(test_block=("test_block", "first"),
                    n_samples=("signed_error_cm", "size"),
                    mean_signed_error_cm=("signed_error_cm", "mean"),
                    mse=("squared_error", "mean")))
    summary["rmse_cm"] = np.sqrt(summary.pop("mse"))
    return summary.assign(**metadata)


def decode(
    task,
    output_dir,
    seed,
    bin_size,
    n_position_bins,
    n_epochs,
    loss_fn,
    decoder_type="both",
    bounds=(0, 200),
):
    """return trial summaries, errors, number of units"""
    path = Path(task["nwb_path"])

    if decoder_type not in {"bayesian", "deepnet", "both"}:
        raise ValueError("decoder_type must be bayesian, deepnet, or both.")
    decoder_names = (
        ("bayesian", "deepnet")
        if decoder_type == "both"
        else (decoder_type,)
    )

    edges = np.linspace(*bounds, n_position_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    base = {key: task[key] for key in ("experiment", "mouse", "day")}
    
    destination = (Path(output_dir) / "VR")
    destination.mkdir(parents=True, exist_ok=True)

    # initialise output dict
    result = {
        "runs": [],
        "seed": seed, # reproduce
    }

    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwb = nap.NWBFile(io.read())
        clusters = nwb["units"]
        n_units = len(clusters)

        behaviour = get_ramps_input(nwb, next(iter(clusters.keys())))
        trials = prepare_trials(nwb, behaviour, bin_size, bounds)

        for trial_type in (0, 1):
            selected = [
                trial for trial in trials
                if trial["trial_type"] == trial_type
            ]
            
            for scheme in ("block0_only", "mixed"):
                context = {
                    **base,
                    'trial_type': trial_type,
                    'scheme': scheme,
                    'n_units': n_units,
                }
                
                try:
                    train_trials, test_trials = split_trials(selected, seed=seed, scheme=scheme)
                except InsufficientTrials as exc:
                    continue
                
                for decoder_name in decoder_names:
                    metadata = {
                        **context,
                        'decoder': decoder_name,
                    }
                    
                    try:
                        if decoder_name == "bayesian":
                            tuning_curves = nap.compute_tuning_curves(
                                clusters,
                                behaviour["P"],
                                bins=edges,
                                epochs=combine_epochs(train_trials),
                            )
                            tuning_curves = tuning_curves.clip(min=1e-8)

                        else: # deepnet
                            fitted = fit_deepnet(
                                train_trials, 
                                edges,
                                loss_fn=loss_fn,
                                n_epochs=n_epochs, 
                                seed=seed,
                            )

                        frames, probabilities = [], []

                        for trial in test_trials:
                            if decoder_name == "bayesian":
                                prediction, probability = nap.decode_bayes(
                                    tuning_curves=tuning_curves,
                                    data=clusters,
                                    epochs=trial["epochs"],
                                    bin_size=bin_size,
                                    sliding_window_size=4, # a bit of smoothing
                                )
                                decoded = np.asarray(prediction.d)
                                probability = np.asarray(probability.d)

                            else: # deepnet
                                decoded, probability = predict_deepnet(
                                    fitted, trial["X"], centres
                                )

                            signed_error = decoded - trial["position"]
                            absolute_error = np.abs(signed_error)

                            frame = pd.DataFrame({
                                "time": trial["time"],
                                "trial": trial["trial"],
                                "trial_type": trial_type,
                                "test_block": trial["block"],
                                "true_position": trial["position"],
                                "decoded_position": decoded,
                                "signed_error_cm": signed_error,
                                "absolute_error_cm": absolute_error,
                            })
                            frames.append(frame)
                            probabilities.append(probability)

                        samples = pd.concat(frames, ignore_index=True)
                        probability = np.concatenate(probabilities)
                        
                        result["runs"].append({
                            'metadata': metadata,
                            'samples': samples,
                            'probability': probability,
                            'train_trials': np.array([
                                trial['trial'] for trial in train_trials
                            ]),
                            'test_trials': np.array([
                                trial['trial'] for trial in test_trials
                            ]),
                            'summary': summarise_trials(samples, metadata),
                        })
                        
                    except Exception:
                        error = {
                            "trial_type": trial_type,
                            "scheme": scheme,
                            "decoder": decoder_name,
                            "traceback": traceback.format_exc(),
                        }
                        print(error["traceback"], flush=True)   
                        
    return result