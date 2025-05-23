"""Module for running round-trip inference and accuracy scoring of backward predictions
using a forward Chemformer model"""

import logging
import subprocess

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.chemformer import Chemformer
from aizynthmodels.chemformer.utils.round_trip_inference import (
    compute_round_trip_accuracy,
    convert_to_input_format,
    create_round_trip_dataset,
    run_test_callbacks,
    set_output_files,
)
from aizynthmodels.utils.configs.chemformer import round_trip_inference  # noqa: F401
from aizynthmodels.utils.hydra import custom_config


@hydra.main(version_base=None, config_name="round_trip_inference")
@custom_config
def main(config: DictConfig) -> None:
    pl.seed_everything(config.seed)

    config, sampled_data_params = create_round_trip_dataset(config)
    chemformer = Chemformer(config)
    set_output_files(config, chemformer)

    logging.info("Running round-trip inference.")
    output = chemformer.predict()

    # Reformat on original shape [n_batches, batch_size, n_beams]
    sampled_smiles, target_smiles = convert_to_input_format(
        output["predictions"],
        output["ground_truth"],
        sampled_data_params,
        config.n_chunks,
    )

    metrics = compute_round_trip_accuracy(chemformer, sampled_smiles, target_smiles)
    run_test_callbacks(chemformer, metrics)

    logging.info(f"Removing temporary file: {sampled_data_params['round_trip_input_data']}")
    subprocess.check_output(["rm", sampled_data_params["round_trip_input_data"]])
    logging.info("Round-trip inference done!")
    return


if __name__ == "__main__":
    main()
