"""Module that featurizes a library for training an filter model"""

import logging
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rxnutils.chem.utils import split_rsmi
from rxnutils.data.batch_utils import read_csv_batch

from aizynthmodels.template_based.utils import get_filename, make_and_save_input_vector
from aizynthmodels.utils.configs.quick_filter import pipelines  # noqa: F401
from aizynthmodels.utils.hydra import custom_config
from aizynthmodels.utils.smiles import seq_rxn_smiles_to_fingerprint, seq_smiles_to_fingerprint


def _make_inputs(
    config: DictConfig,
    dataset: pd.DataFrame,
    batch: Tuple[int, int, int] = None,
) -> None:
    logging.info("Generating inputs...")
    smiles_column = config.library_columns.reaction_smiles

    products = []
    reactants = []
    for smiles in dataset[smiles_column]:
        reactant, _, product = split_rsmi(smiles)
        reactants.append(reactant)
        products.append(product)

    fp_kwargs = {
        "fp_length": config.model_hyperparams.fingerprint_size,
        "fp_radius": config.model_hyperparams.fingerprint_radius,
        "chirality": config.model_hyperparams.chirality,
    }

    make_and_save_input_vector(
        [products, reactants], seq_rxn_smiles_to_fingerprint, fp_kwargs, batch, get_filename(config, "model_inputs_rxn")
    )
    make_and_save_input_vector(
        [products], seq_smiles_to_fingerprint, fp_kwargs, batch, get_filename(config, "model_inputs_prod")
    )


def _make_labels(
    config: DictConfig,
    dataset: pd.DataFrame,
    batch: Tuple[int, int, int] = None,
) -> None:
    logging.info("Generating labels...")
    labels = dataset[config.library_columns.label].to_numpy()
    filename = get_filename(config, "model_labels")
    if batch is not None:
        filename = filename.replace(".npz", f".{batch[0]}.npz")
    np.savez(filename, labels, compressed=True)


@hydra.main(version_base=None, config_name="filter_pipeline")
@custom_config
def main(config: DictConfig) -> None:
    """Command-line interface for the featurization tool"""
    if config.batch is not None:
        batch = tuple(config.batch)
    else:
        batch = None

    dataset = read_csv_batch(
        get_filename(config, "library"),
        batch=batch,
        sep="\t",
        usecols=[config.library_columns.reaction_smiles, config.library_columns.label],
    )

    _make_inputs(config, dataset, batch)
    _make_labels(config, dataset, batch)


if __name__ == "__main__":
    main()
