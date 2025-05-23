import logging
import tempfile
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.chemformer import Chemformer
from aizynthmodels.utils.type_utils import StrDict


def batchify(smiles_lst: Union[List[str], np.ndarray], batch_size: int) -> Union[List[List[str]], List[np.ndarray]]:
    """
    Create batches given an input list of SMILES or list of list of SMILES.

    :param smiles_list: list of SMILES
    :param batch_size: number of samples in batch
    :return: batched SMILES in a list
    """
    n_samples = len(smiles_lst)
    n_batches = int(np.ceil(n_samples / batch_size))

    batched_smiles = []
    for i_batch in range(n_batches):
        if i_batch != n_batches - 1:
            batched_smiles.append(smiles_lst[i_batch * batch_size : (i_batch + 1) * batch_size])  # noqa: E203
        else:
            batched_smiles.append(smiles_lst[i_batch * batch_size : :])  # noqa: E203
    return batched_smiles


def compute_round_trip_accuracy(
    chemformer: Chemformer,
    sampled_smiles: List[List[str]],
    target_smiles: List[List[str]],
) -> List[Dict[str, Any]]:
    """
    Calculating (round-trip) accuracy given sampled and target SMILES (products).

    :param chemformer: a Chemformer model with a decoder sampler
    :param sampled_smiles: product SMILES sampled by forward Chemformer
    :param target_smiles: ground truth product SMILES
    :return: Dictionary with scores and (batched) predictions
    """
    logging.info("Evaluating predictions.")

    metrics_out = []
    batch_idx = 0
    for sampled_batch, target_batch in zip(sampled_smiles, target_smiles):
        metrics = chemformer.scores.apply(sampled_batch, target_batch, is_canonical=False)

        metrics.update({"sampled_molecules": sampled_batch, "target_smiles": target_batch})

        metrics_out.append(metrics)
        batch_idx += 1
    return metrics_out


def convert_to_input_format(
    sampled_smiles: List[List[str]],
    target_smiles: List[List[str]],
    sampled_data_params: Dict[str, Any],
    n_chunks: int = 1,
) -> Tuple[List[np.ndarray], List[List[str]]]:
    """
    Converting sampled data to original input format such that,
    sampled_smiles: [n_batches, batch_size, n_beams],
    target_smiles: [n_batches, batch_size, 1].

    :param sampled_smiles: SMILES sampled in round-trip inference
    :param target_smiles: target SMILES (ground truth product)
    :param sampled_data_params: parameters of the input data from backward predictions
            (batch_size, beam_size, n_samples)
    :return: Reshaped round-trip predictions.
    """
    batch_size = sampled_data_params["batch_size"]
    n_beams = sampled_data_params["beam_size"]
    n_samples = sampled_data_params["n_samples"]

    sampled_smiles = np.array(sampled_smiles)
    target_smiles = np.array(target_smiles)

    sampled_smiles = np.reshape(sampled_smiles, (-1, n_beams))
    target_smiles = np.reshape(target_smiles, (-1, n_beams))

    if n_chunks == 1:
        assert target_smiles.shape[0] == n_samples

    # Sanity-check that target smiles are the same within beams
    for tgt_beams in target_smiles:
        assert np.all(tgt_beams == tgt_beams[0])

    # Extract the target smiles for each original sample
    target_smiles = [tgt_smi[0] for tgt_smi in target_smiles]

    smpl_smiles_reform = batchify(sampled_smiles, batch_size)
    tgt_smiles_reform = batchify(target_smiles, batch_size)

    return smpl_smiles_reform, tgt_smiles_reform


def create_round_trip_dataset(config: DictConfig) -> Tuple[DictConfig, StrDict]:
    """
    Reading sampled smiles and creating dataframe on synthesis-datamodule format.

    :param config: Input arguments with parameters for Chemformer, data paths etc.
    :return: Updated arguments and input-data metadata dictionary
    """
    logging.info("Creating input data from sampled predictions.")

    _, round_trip_input_data = tempfile.mkstemp(suffix=".csv")

    input_data = pd.read_csv(config.input_data, sep="\t")
    input_data = input_data.iloc[input_data["set"].values == config.dataset_part]

    target_column = config.target_column

    input_targets = input_data[target_column].values

    predicted_data = pd.read_json(config.backward_predictions, orient="table")

    batch_size = len(predicted_data["sampled_molecules"].values[0])
    n_samples = sum([len(batch_smiles) for batch_smiles in predicted_data["sampled_molecules"].values])
    n_beams = len(predicted_data["sampled_molecules"].values[0][0])

    sampled_data_params = {
        "n_samples": n_samples,
        "beam_size": n_beams,
        "batch_size": batch_size,
        "round_trip_input_data": round_trip_input_data,
    }

    counter = 0
    sampled_smiles = []
    target_smiles = []
    # Unravel predictions
    for batch_smiles in predicted_data["sampled_molecules"].values:
        for top_n_smiles in batch_smiles:
            sampled_smiles.extend(top_n_smiles)
            target_smiles.extend([input_targets[counter] for _ in range(n_beams)])
            counter += 1

    input_data = pd.DataFrame(
        {
            "reactants": sampled_smiles,
            "products": target_smiles,
            "set": len(target_smiles) * ["test"],
        }
    )

    logging.info(f"Writing data to temporary file: {round_trip_input_data}")
    input_data.to_csv(round_trip_input_data, sep="\t", index=False)

    config.data_path = round_trip_input_data
    return config, sampled_data_params


def set_output_files(args, chemformer):
    if args.output_score_data or args.output_predictions:
        for callback in chemformer.trainer.callbacks:
            if hasattr(callback, "set_output_files"):
                callback.set_output_files(args.output_score_data, args.output_predictions)


def run_test_callbacks(chemformer: Chemformer, metrics_scores: List[Dict[str, Any]]) -> None:
    """Run callback.on_test_batch_end on all (scoring) callbacks."""
    for batch_idx, scores in enumerate(metrics_scores):
        for callback in chemformer.trainer.callbacks:
            if not isinstance(callback, pl.callbacks.progress.ProgressBar):
                callback.on_test_batch_end(chemformer.trainer, chemformer.model, scores, {}, batch_idx, 0)
